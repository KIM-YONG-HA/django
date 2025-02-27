from django import template

import markdown
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter
def sub(value, arg):
    return value - arg

@register.filter
def mark(value):
    extentions = ['nl2br', 'fenced_code']
    return mark_safe(markdown.markdown(value, extensions=extentions))



