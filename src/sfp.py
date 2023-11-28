from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from enum import Enum
from warnings import warn

from fastapi import Depends, Query
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Extra, ValidationError, create_model, fields as py_field, validator, Field
from pydantic.fields import FieldInfo
from sqlalchemy import or_
import sqlalchemy.orm as orm
from sqlalchemy.sql.selectable import Select
from sqlalchemy.sql.elements import BinaryExpression


def _backward_compatible_value_for_like_and_ilike(value: str):
    """Add % if not in value to be backward compatible.

    Args:
        value (str): The value to filter.

    Returns:
        Either the unmodified value if a percent sign is present, the value wrapped in % otherwise to preserve
        current behavior.
    """
    if "%" not in value:
        warn(
            "You must pass the % character explicitly to use the like and ilike operators.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = f"%{value}%"
    return value


_orm_operator_transformer = {
    "neq": lambda value: ("__ne__", value),
    "gt": lambda value: ("__gt__", value),
    "gte": lambda value: ("__ge__", value),
    "in": lambda value: ("in_", value),
    "isnull": lambda value: ("is_", None) if value is True else ("is_not", None),
    "lt": lambda value: ("__lt__", value),
    "lte": lambda value: ("__le__", value),
    "like": lambda value: ("like", _backward_compatible_value_for_like_and_ilike(value)),
    "ilike": lambda value: ("ilike", _backward_compatible_value_for_like_and_ilike(value)),
    # XXX(arthurio): Mysql excludes None values when using `in` or `not in` filters.
    "not": lambda value: ("is_not", value),
    "not_in": lambda value: ("not_in", value),
}


class Filter(BaseModel, extra=Extra.forbid):
    class Direction(str, Enum):
        asc = "asc"
        desc = "desc"

    class Constants:  # pragma: no cover
        model: Type
        ordering_field_name: str = "order_by"
        search_model_fields: List[str]
        search_field_name: str = "search"
        prefix: Optional[str] = None

    @property
    def filtering_fields(self):
        fields = self.dict(exclude_none=True, exclude_unset=True)
        fields.pop(self.Constants.ordering_field_name, None)
        fields.pop(self.Constants.search_field_name, None)

        return fields.items()

    @property
    def ordering_fields(self):
        fields = self.dict(exclude_none=True, exclude_unset=True)

        for i in list(fields.keys()):
            if not isinstance(getattr(self, i), Filter):
                fields.pop(i, None)

        return fields.items()

    @validator("*", pre=True)
    def split_str(cls, value, field):
        if ((field.name == cls.Constants.ordering_field_name
             or field.name.endswith("__in")
             or field.name.endswith("__not_in")) and isinstance(value, str)):
            return [field.type_(v) for v in value.split(",")]
        return value

    @validator("*", pre=True, allow_reuse=True, check_fields=False)
    def strip_order_by_values(cls, value, values, field):
        if field.name != cls.Constants.ordering_field_name:
            return value

        if not value:
            return None

        stripped_values = []
        for field_name in value:
            stripped_value = field_name.strip()
            if stripped_value:
                stripped_values.append(stripped_value)

        return stripped_values

    @classmethod
    def _split_prefix(cls, field_name: str) -> Tuple[Optional[str], str]:
        field_split = field_name.split('__', 1)

        if len(field_split) == 1:
            return None, field_split[0]
        else:
            return field_split[0], field_split[1]

    @classmethod
    def _validate_order_by_field(cls, ordering_field: str, fields_dict: Dict) -> bool:
        prefix, field_name = cls._split_prefix(ordering_field)

        for key in fields_dict.keys():
            value = fields_dict.get(key)

            if isinstance(value, Filter):
                if (hasattr(value.Constants.model, field_name) and value.Constants.prefix == prefix) or \
                        cls._validate_order_by_field(ordering_field, dict(value)):
                    return True

        return False

    @validator(Constants.ordering_field_name, allow_reuse=True, check_fields=False)
    def validate_order_by_field(cls, value, values, field):
        field_name_usages = defaultdict(list)
        duplicated_field_names = set()

        for field_name_with_direction in value:
            field_name = field_name_with_direction.replace("-", "").replace("+", "")

            if not hasattr(cls.Constants.model, field_name) and not cls._validate_order_by_field(field_name, values):
                raise ValueError(f"{field_name} is not a valid ordering field.")

            field_name_usages[field_name].append(field_name_with_direction)
            if len(field_name_usages[field_name]) > 1:
                duplicated_field_names.add(field_name)

        if duplicated_field_names:
            ambiguous_field_names = ", ".join(
                [
                    field_name_with_direction
                    for field_name in sorted(duplicated_field_names)
                    for field_name_with_direction in field_name_usages[field_name]
                ]
            )
            raise ValueError(
                f"Field names can appear at most once for {cls.Constants.ordering_field_name}. "
                f"The following was ambiguous: {ambiguous_field_names}."
            )

        return value

    def filter(self, query: Union[orm.Query, Select], search_filters: List[BinaryExpression], search_value: str):
        for field_name, value in self.filtering_fields:
            field_value = getattr(self, field_name)
            if isinstance(field_value, Filter):
                query = field_value.filter(query, search_filters, search_value)
            else:
                if "__" in field_name:
                    field_name, operator = field_name.split("__")
                    operator, value = _orm_operator_transformer[operator](value)
                else:
                    operator = "__eq__"

                model_field = getattr(self.Constants.model, field_name)
                query = query.filter(getattr(model_field, operator)(value))

        if hasattr(self.Constants, "search_model_fields") and search_value is not None:
            search_filters.extend([
                getattr(self.Constants.model, field).ilike(f"%{search_value}%")
                for field in self.Constants.search_model_fields
            ])

        return query

    def sort(self, query: Union[orm.Query, Select], ordering_fields: List[str]):
        if len(ordering_fields) == 0:
            return query

        for field_full_name in ordering_fields:
            prefix, field_name = Filter._split_prefix(field_full_name.replace("-", "").replace("+", ""))
            direction = Filter.Direction.asc

            if field_full_name.startswith("-"):
                direction = Filter.Direction.desc

            if hasattr(self.Constants.model, field_name) and \
                    (self.Constants.prefix == prefix or prefix is None):

                order_by_field = getattr(self.Constants.model, field_name)
                query = query.order_by(getattr(order_by_field, direction)())
            else:
                for ordering_fields_name, _ in self.ordering_fields:
                    field_value = getattr(self, ordering_fields_name)

                    if isinstance(field_value, Filter):
                        query = field_value.sort(query, [field_full_name])

        return query


def with_prefix(entry: Type[Filter]):
    class NestedFilter(entry):  # type: ignore[misc, valid-type]
        class Config:
            extra = Extra.forbid

            @classmethod
            def alias_generator(cls, string: str) -> str:
                return f"{entry.Constants.prefix}__{string}"

        class Constants(entry.Constants):  # type: ignore[name-defined]
            ...

    return NestedFilter


def _list_to_str_fields(entry: Type[Filter], discard_ordering_field, discard_search_field):
    ret: Dict[str, Tuple[Union[object, Type], Optional[FieldInfo]]] = {}
    for f in entry.__fields__.values():
        field_info = deepcopy(f.field_info)

        if (discard_ordering_field and entry.Constants.ordering_field_name == f.name) or \
                (discard_search_field and entry.Constants.search_field_name == f.name):
            continue

        if f.shape == py_field.SHAPE_LIST:
            if isinstance(field_info.default, Iterable):
                field_info.default = ",".join(field_info.default)

            ret[f.name] = (str if f.required else Optional[str], field_info)
        else:
            field_type = entry.__annotations__.get(f.name, f.outer_type_)

            ret[f.name] = (field_type if f.required else Optional[field_type], field_info)

    return ret


def FilterDepends(entry: Type[Filter], *, by_alias: bool = False, discard_ordering_field=False,
                  discard_search_field=False) -> Any:
    """Use a hack to support lists in filters.

    FastAPI doesn't support it yet: https://github.com/tiangolo/fastapi/issues/50

    What we do is loop through the fields of a filter and change any `list` field to a `str` one so that it won't be
    excluded from the possible query parameters.

    When we apply the filter, we build the original filter to properly validate the data (i.e. can the string be parsed
    and formatted as a list of <type>?)
    """
    fields = _list_to_str_fields(entry, discard_ordering_field, discard_search_field)
    GeneratedFilter: Type[Filter] = create_model(entry.__class__.__name__, **fields)

    class FilterWrapper(GeneratedFilter):  # type: ignore[misc,valid-type]
        def filter(self, *args, **kwargs):
            try:
                original_filter = entry(**self.dict(by_alias=by_alias))
            except ValidationError as e:
                raise RequestValidationError(e.raw_errors) from e

            search_filters = []

            search_value = getattr(self, entry.Constants.search_field_name)
            query = original_filter.filter(*args, search_filters, search_value, **kwargs)
            query = query.filter(or_(*search_filters))

            return query

        def sort(self, *args, **kwargs):
            try:
                original_filter = entry(**self.dict(by_alias=by_alias))
            except ValidationError as e:
                raise RequestValidationError(e.raw_errors) from e

            ordering_fields = []

            if getattr(self, entry.Constants.ordering_field_name) is not None:
                ordering_fields = "".join(getattr(self, entry.Constants.ordering_field_name).split()).split(',')

            return original_filter.sort(*args, ordering_fields, **kwargs)

    return Depends(FilterWrapper)
