from collections import deque

import pytest

from tprofiler.utils import Stack


@pytest.fixture
def empty_stack():
    return Stack()


@pytest.fixture
def stack_with_items():
    stack = Stack()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    return stack


@pytest.fixture
def single_item_stack():
    stack = Stack()
    stack.push("test")
    return stack


@pytest.mark.unittest
class TestStack:
    def test_init(self, empty_stack):
        assert isinstance(empty_stack._items, deque)
        assert empty_stack.is_empty
        assert empty_stack.size == 0

    def test_push_single_item(self, empty_stack):
        empty_stack.push(1)
        assert not empty_stack.is_empty
        assert empty_stack.size == 1
        assert empty_stack.peek() == 1

    def test_push_multiple_items(self, empty_stack):
        items = [1, 2, 3, "test", None]
        for item in items:
            empty_stack.push(item)

        assert empty_stack.size == len(items)
        assert empty_stack.peek() == None

    def test_pop_from_non_empty_stack(self, stack_with_items):
        assert stack_with_items.pop() == 3
        assert stack_with_items.size == 2
        assert stack_with_items.peek() == 2

    def test_pop_from_empty_stack(self, empty_stack):
        with pytest.raises(IndexError, match="pop from empty stack"):
            empty_stack.pop()

    def test_pop_until_empty(self, single_item_stack):
        assert single_item_stack.pop() == "test"
        assert single_item_stack.is_empty

        with pytest.raises(IndexError, match="pop from empty stack"):
            single_item_stack.pop()

    def test_peek_non_empty_stack(self, stack_with_items):
        original_size = stack_with_items.size
        assert stack_with_items.peek() == 3
        assert stack_with_items.size == original_size  # peek shouldn't change size

    def test_peek_empty_stack(self, empty_stack):
        with pytest.raises(IndexError, match="peek from empty stack"):
            empty_stack.peek()

    def test_clear_empty_stack(self, empty_stack):
        empty_stack.clear()
        assert empty_stack.is_empty
        assert empty_stack.size == 0

    def test_clear_non_empty_stack(self, stack_with_items):
        stack_with_items.clear()
        assert stack_with_items.is_empty
        assert stack_with_items.size == 0

    def test_is_empty_property_true(self, empty_stack):
        assert empty_stack.is_empty is True

    def test_is_empty_property_false(self, stack_with_items):
        assert stack_with_items.is_empty is False

    def test_size_property_empty(self, empty_stack):
        assert empty_stack.size == 0

    def test_size_property_non_empty(self, stack_with_items):
        assert stack_with_items.size == 3

    def test_str_empty_stack(self, empty_stack):
        assert str(empty_stack) == "Stack([])"

    def test_str_non_empty_stack(self, stack_with_items):
        assert str(stack_with_items) == "Stack([1, 2, 3])"

    def test_len_empty_stack(self, empty_stack):
        assert len(empty_stack) == 0

    def test_len_non_empty_stack(self, stack_with_items):
        assert len(stack_with_items) == 3

    def test_bool_empty_stack(self, empty_stack):
        assert bool(empty_stack) is False
        assert not empty_stack

    def test_bool_non_empty_stack(self, stack_with_items):
        assert bool(stack_with_items) is True
        assert stack_with_items

    def test_operations_sequence(self, empty_stack):
        # Test a sequence of operations
        assert empty_stack.is_empty

        empty_stack.push(1)
        empty_stack.push(2)
        assert empty_stack.size == 2
        assert empty_stack.peek() == 2

        popped = empty_stack.pop()
        assert popped == 2
        assert empty_stack.size == 1
        assert empty_stack.peek() == 1

        empty_stack.push(3)
        assert empty_stack.size == 2

        empty_stack.clear()
        assert empty_stack.is_empty
