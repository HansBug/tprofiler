"""
A high-performance stack implementation using deque.

This module provides a Stack class that implements the Last-In-First-Out (LIFO) 
data structure using Python's deque for optimal performance. The stack supports 
all standard stack operations with O(1) time complexity.
"""

from collections import deque


class Stack:
    """
    A high-performance stack implementation using deque.

    This class implements a Last-In-First-Out (LIFO) data structure using 
    Python's deque for better performance compared to list-based implementations.
    All basic stack operations (push, pop, peek) have O(1) time complexity.

    Example::

        >>> stack = Stack()
        >>> stack.push(1)
        >>> stack.push(2)
        >>> stack.peek()
        2
        >>> stack.pop()
        2
        >>> stack.size
        1
    """

    def __init__(self):
        """
        Initialize an empty stack.

        Uses deque implementation for better performance with O(1) operations.
        """
        self._items = deque()

    def push(self, item):
        """
        Push an item onto the top of the stack.

        :param item: The item to push onto the stack.
        :type item: Any

        Time complexity: O(1)

        Example::

            >>> stack = Stack()
            >>> stack.push(42)
            >>> stack.size
            1
        """
        self._items.append(item)

    def pop(self):
        """
        Remove and return the top item from the stack.

        :return: The top item from the stack.
        :rtype: Any
        :raises IndexError: If the stack is empty.

        Time complexity: O(1)

        Example::

            >>> stack = Stack()
            >>> stack.push(42)
            >>> stack.pop()
            42
        """
        if self.is_empty:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        """
        Return the top item from the stack without removing it.

        :return: The top item from the stack.
        :rtype: Any
        :raises IndexError: If the stack is empty.

        Time complexity: O(1)

        Example::

            >>> stack = Stack()
            >>> stack.push(42)
            >>> stack.peek()
            42
            >>> stack.size
            1
        """
        if self.is_empty:
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def clear(self):
        """
        Remove all items from the stack.

        After calling this method, the stack will be empty.

        Example::

            >>> stack = Stack()
            >>> stack.push(1)
            >>> stack.push(2)
            >>> stack.clear()
            >>> stack.is_empty
            True
        """
        self._items.clear()

    @property
    def is_empty(self):
        """
        Check if the stack is empty.

        :return: True if the stack is empty, False otherwise.
        :rtype: bool

        Time complexity: O(1)

        Example::

            >>> stack = Stack()
            >>> stack.is_empty
            True
            >>> stack.push(1)
            >>> stack.is_empty
            False
        """
        return len(self._items) == 0

    @property
    def size(self):
        """
        Get the number of items in the stack.

        :return: The number of items in the stack.
        :rtype: int

        Time complexity: O(1)

        Example::

            >>> stack = Stack()
            >>> stack.size
            0
            >>> stack.push(1)
            >>> stack.push(2)
            >>> stack.size
            2
        """
        return len(self._items)

    def __str__(self):
        """
        Return a string representation of the stack.

        :return: String representation showing the stack contents.
        :rtype: str

        Example::

            >>> stack = Stack()
            >>> stack.push(1)
            >>> stack.push(2)
            >>> str(stack)
            'Stack([1, 2])'
        """
        return f"Stack({list(self._items)})"

    def __len__(self):
        """
        Support for the len() function.

        :return: The number of items in the stack.
        :rtype: int

        Example::

            >>> stack = Stack()
            >>> stack.push(1)
            >>> stack.push(2)
            >>> len(stack)
            2
        """
        return len(self._items)

    def __bool__(self):
        """
        Support for boolean value judgment.

        :return: True if the stack is not empty, False if empty.
        :rtype: bool

        Example::

            >>> stack = Stack()
            >>> bool(stack)
            False
            >>> stack.push(1)
            >>> bool(stack)
            True
        """
        return not self.is_empty
