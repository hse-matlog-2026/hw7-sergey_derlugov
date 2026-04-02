# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2025
# File name: predicates/syntax.py

"""Syntactic handling of predicate-logic expressions."""

from __future__ import annotations
from functools import lru_cache
from typing import AbstractSet, Mapping, Optional, Sequence, Set, Tuple, Union

from logic_utils import fresh_variable_name_generator, frozen, \
                        memoized_parameterless_method

from propositions.syntax import Formula as PropositionalFormula, \
                                is_variable as is_propositional_variable

class ForbiddenVariableError(Exception):
    """Raised by `Term.substitute` and `Formula.substitute` when a substituted
    term contains a variable name that is forbidden in that context.

    Attributes:
        variable_name (`str`): the variable name that was forbidden in the
            context in which a term containing it was to be substituted.
    """
    variable_name: str

    def __init__(self, variable_name: str):
        """Initializes a `ForbiddenVariableError` from the offending variable
        name.

        Parameters:
            variable_name: variable name that is forbidden in the context in
                which a term containing it is to be substituted.
        """
        assert is_variable(variable_name)
        self.variable_name = variable_name

@lru_cache(maxsize=100) # Cache the return value of is_constant
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant name, ``False`` otherwise.
    """
    return  (((string[0] >= '0' and string[0] <= '9') or \
              (string[0] >= 'a' and string[0] <= 'e')) and \
             string.isalnum()) or string == '_'

@lru_cache(maxsize=100) # Cache the return value of is_variable
def is_variable(string: str) -> bool:
    """Checks if the given string is a variable name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a variable name, ``False`` otherwise.
    """
    return string[0] >= 'u' and string[0] <= 'z' and string.isalnum()

@lru_cache(maxsize=100) # Cache the return value of is_function
def is_function(string: str) -> bool:
    """Checks if the given string is a function name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a function name, ``False`` otherwise.
    """
    return string[0] >= 'f' and string[0] <= 't' and string.isalnum()

def _parse_term_name(s: str) -> Tuple[str, str]:
    """Reads a maximal term-root name (constant, variable, or function name)."""
    if not s:
        raise ValueError('Unexpected end of input while parsing term name')
    if s[0] == '_':
        return '_', s[1:]
    c = s[0]
    if c.isdigit() or ('a' <= c <= 'e'):
        i = 1
        while i < len(s) and s[i].isalnum():
            i += 1
        return s[:i], s[i:]
    if 'u' <= c <= 'z':
        i = 1
        while i < len(s) and s[i].isalnum():
            i += 1
        return s[:i], s[i:]
    if 'f' <= c <= 't':
        i = 1
        while i < len(s) and s[i].isalnum():
            i += 1
        return s[:i], s[i:]
    raise ValueError('Invalid term start')

def _parse_relation_name(s: str) -> Tuple[str, str]:
    """Reads a maximal relation name (``F``..``T`` prefix)."""
    if not s or not ('F' <= s[0] <= 'T'):
        raise ValueError('Invalid relation name')
    i = 1
    while i < len(s) and s[i].isalnum():
        i += 1
    return s[:i], s[i:]

_BINARY_OP_INFO = {
    '->': (10, 'right'),
    '|': (20, 'left'),
    '&': (30, 'left'),
}

@frozen
class Term:
    """An immutable predicate-logic term in tree representation, composed from
    variable names and constant names, and function names applied to them.

    Attributes:
        root (`str`): the constant name, variable name, or function name at the
            root of the term tree.
        arguments (`~typing.Optional`\\[`~typing.Tuple`\\[`Term`, ...]]): the
            arguments of the root, if the root is a function name.
    """
    root: str
    arguments: Optional[Tuple[Term, ...]]

    def __init__(self, root: str, arguments: Optional[Sequence[Term]] = None):
        """Initializes a `Term` from its root and root arguments.

        Parameters:
            root: the root for the formula tree.
            arguments: the arguments for the root, if the root is a function
                name.
        """
        if is_constant(root) or is_variable(root):
            assert arguments is None
            self.root = root
            self.arguments = None
        else:
            assert is_function(root)
            assert arguments is not None and len(arguments) > 0
            self.root = root
            self.arguments = tuple(arguments)

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current term.

        Returns:
            The standard string representation of the current term.
        """
        if is_constant(self.root) or is_variable(self.root):
            return self.root
        return self.root + '(' + ','.join(str(a) for a in self.arguments) + ')'

    def __eq__(self, other: object) -> bool:
        """Compares the current term with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Term` object that equals the
            current term, ``False`` otherwise.
        """
        return isinstance(other, Term) and str(self) == str(other)
        
    def __ne__(self, other: object) -> bool:
        """Compares the current term with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Term` object or does not
            equal the current term, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Term, str]:
        """Parses a prefix of the given string into a term.

        Parameters:
            string: string to parse, which has a prefix that is a valid
                representation of a term.

        Returns:
            A pair of the parsed term and the unparsed suffix of the string. If
            the given string has as a prefix a constant name (e.g., ``'c12'``)
            or a variable name (e.g., ``'x12'``), then the parsed prefix will be
            that entire name (and not just a part of it, such as ``'x1'``).
        """
        name, remainder = _parse_term_name(string)
        if is_function(name):
            if not remainder or remainder[0] != '(':
                raise ValueError('Expected ( after function name')
            remainder = remainder[1:]
            args: list = []
            if remainder and remainder[0] == ')':
                raise ValueError('Function must have positive arity')
            while True:
                arg, remainder = Term._parse_prefix(remainder)
                args.append(arg)
                if not remainder:
                    raise ValueError('Expected ) or , in argument list')
                if remainder[0] == ')':
                    return Term(name, tuple(args)), remainder[1:]
                if remainder[0] != ',':
                    raise ValueError('Expected , or )')
                remainder = remainder[1:]
        assert is_constant(name) or is_variable(name)
        return Term(name), remainder

    @staticmethod
    def parse(string: str) -> Term:
        """Parses the given valid string representation into a term.

        Parameters:
            string: string to parse.

        Returns:
            A term whose standard string representation is the given string.
        """
        term, r = Term._parse_prefix(string)
        assert r == '', 'Incomplete parse of term'
        return term

    def constants(self) -> Set[str]:
        """Finds all constant names in the current term.

        Returns:
            A set of all constant names used in the current term.
        """
        if is_constant(self.root):
            return {self.root}
        if is_variable(self.root):
            return set()
        result: Set[str] = set()
        for a in self.arguments:
            result |= a.constants()
        return result

    def variables(self) -> Set[str]:
        """Finds all variable names in the current term.

        Returns:
            A set of all variable names used in the current term.
        """
        if is_variable(self.root):
            return {self.root}
        if is_constant(self.root):
            return set()
        result: Set[str] = set()
        for a in self.arguments:
            result |= a.variables()
        return result

    def functions(self) -> Set[Tuple[str, int]]:
        """Finds all function names in the current term, along with their
        arities.

        Returns:
            A set of pairs of function name and arity (number of arguments) for
            all function names used in the current term.
        """
        if is_constant(self.root) or is_variable(self.root):
            return set()
        result: Set[Tuple[str, int]] = {(self.root, len(self.arguments))}
        for a in self.arguments:
            result |= a.functions()
        return result

    def substitute(self, substitution_map: Mapping[str, Term],
                   forbidden_variables: AbstractSet[str] = frozenset()) -> Term:
        """Substitutes in the current term, each constant name `construct` or
        variable name `construct` that is a key in `substitution_map` with the
        term `substitution_map`\\ ``[``\\ `construct`\\ ``]``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.
            forbidden_variables: variable names not allowed in substitution
                terms.

        Returns:
            The term resulting from performing all substitutions. Only
            constant name and variable name occurrences originating in the
            current term are substituted (i.e., those originating in one of the
            specified substitutions are not subjected to additional
            substitutions).

        Raises:
            ForbiddenVariableError: If a term that is used in the requested
                substitution contains a variable name from
                `forbidden_variables`.

        Examples:
            >>> Term.parse('f(x,c)').substitute(
            ...     {'c': Term.parse('plus(d,x)'), 'x': Term.parse('c')}, {'y'})
            f(c,plus(d,x))

            >>> Term.parse('f(x,c)').substitute(
            ...     {'c': Term.parse('plus(d,y)')}, {'y'})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: y
        """
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)
        # Task 9.1

@lru_cache(maxsize=100) # Cache the return value of is_equality
def is_equality(string: str) -> bool:
    """Checks if the given string is the equality relation.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is the equality relation, ``False``
        otherwise.
    """
    return string == '='

@lru_cache(maxsize=100) # Cache the return value of is_relation
def is_relation(string: str) -> bool:
    """Checks if the given string is a relation name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a relation name, ``False`` otherwise.
    """
    return string[0] >= 'F' and string[0] <= 'T' and string.isalnum()

@lru_cache(maxsize=100) # Cache the return value of is_unary
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == '~'

@lru_cache(maxsize=100) # Cache the return value of is_binary
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return string == '&' or string == '|' or string == '->'

@lru_cache(maxsize=100) # Cache the return value of is_quantifier
def is_quantifier(string: str) -> bool:
    """Checks if the given string is a quantifier.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a quantifier, ``False`` otherwise.
    """
    return string == 'A' or string == 'E'

@frozen
class Formula:
    """An immutable predicate-logic formula in tree representation, composed
    from relation names applied to predicate-logic terms, and operators and
    quantifications applied to them.

    Attributes:
        root (`str`): the relation name, equality relation, operator, or
            quantifier at the root of the formula tree.
        arguments (`~typing.Optional`\\[`~typing.Tuple`\\[`Term`, ...]]): the
            arguments of the root, if the root is a relation name or the
            equality relation.
        first (`~typing.Optional`\\[`Formula`]): the first operand of the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second operand of the
            root, if the root is a binary operator.
        variable (`~typing.Optional`\\[`str`]): the variable name quantified by
            the root, if the root is a quantification.
        statement (`~typing.Optional`\\[`Formula`]): the statement quantified by
            the root, if the root is a quantification.
    """
    root: str
    arguments: Optional[Tuple[Term, ...]]
    first: Optional[Formula]
    second: Optional[Formula]
    variable: Optional[str]
    statement: Optional[Formula]

    def __init__(self, root: str,
                 arguments_or_first_or_variable: Union[Sequence[Term],
                                                       Formula, str],
                 second_or_statement: Optional[Formula] = None):
        """Initializes a `Formula` from its root and root arguments, root
        operands, or root quantified variable name and statement.

        Parameters:
            root: the root for the formula tree.
            arguments_or_first_or_variable: the arguments for the root, if the
                root is a relation name or the equality relation; the first
                operand for the root, if the root is a unary or binary operator;
                the variable name to be quantified by the root, if the root is a
                quantification.
            second_or_statement: the second operand for the root, if the root is
                a binary operator; the statement to be quantified by the root,
                if the root is a quantification.
        """
        if is_equality(root) or is_relation(root):
            # Populate self.root and self.arguments
            assert isinstance(arguments_or_first_or_variable, Sequence) and \
                   not isinstance(arguments_or_first_or_variable, str)
            if is_equality(root):
                assert len(arguments_or_first_or_variable) == 2
            assert second_or_statement is None
            self.root, self.arguments = \
                root, tuple(arguments_or_first_or_variable)
        elif is_unary(root):
            # Populate self.first
            assert isinstance(arguments_or_first_or_variable, Formula)
            assert second_or_statement is None
            self.root, self.first = root, arguments_or_first_or_variable
        elif is_binary(root):
            # Populate self.first and self.second
            assert isinstance(arguments_or_first_or_variable, Formula)
            assert second_or_statement is not None
            self.root, self.first, self.second = \
                root, arguments_or_first_or_variable, second_or_statement
        else:
            assert is_quantifier(root)
            # Populate self.variable and self.statement
            assert isinstance(arguments_or_first_or_variable, str) and \
                   is_variable(arguments_or_first_or_variable)
            assert second_or_statement is not None
            self.root, self.variable, self.statement = \
                root, arguments_or_first_or_variable, second_or_statement

    @staticmethod
    def _parse_binary(string: str, min_precedence: int) -> Tuple['Formula', str]:
        first, s = Formula._parse_unary_full(string)
        while s:
            op = None
            for cand in ('->', '|', '&'):
                if s.startswith(cand):
                    op = cand
                    break
            if op is None:
                break
            prec, assoc = _BINARY_OP_INFO[op]
            if prec < min_precedence:
                break
            s = s[len(op):]
            rhs_min = prec if assoc == 'right' else prec + 1
            second, s = Formula._parse_binary(s, rhs_min)
            first = Formula(op, first, second)
        return first, s

    @staticmethod
    def _parse_unary_full(string: str) -> Tuple['Formula', str]:
        """Unary operators with a fully-parsed operand (may contain binary ops)."""
        if string.startswith('~'):
            inner, rest = Formula._parse_unary_full(string[1:])
            return Formula('~', inner), rest
        return Formula._parse_atomic_full(string)

    @staticmethod
    def _parse_atomic_full(string: str) -> Tuple['Formula', str]:
        """Relation, equality, quantifier, or parenthesized subformula (full)."""
        return Formula._parse_atomic_impl(string)

    @staticmethod
    def _parse_prefix_unary(string: str) -> Tuple['Formula', str]:
        """Unary chain, then one atomic piece without absorbing a top-level |, &, ->."""
        if string.startswith('~'):
            inner, rest = Formula._parse_prefix_unary(string[1:])
            return Formula('~', inner), rest
        return Formula._parse_atomic_impl(string)

    @staticmethod
    def _parse_atomic_impl(string: str) -> Tuple['Formula', str]:
        if not string:
            raise ValueError('Unexpected end of input in formula')
        if string[0] in 'AE' and len(string) > 1 and 'u' <= string[1] <= 'z':
            quantifier = string[0]
            var, rest = _parse_term_name(string[1:])
            assert is_variable(var)
            if not rest.startswith('['):
                raise ValueError('Expected [ after quantified variable')
            depth = 1
            i = 1
            while depth > 0:
                if i >= len(rest):
                    raise ValueError('Unbalanced brackets in quantified formula')
                if rest[i] == '[':
                    depth += 1
                elif rest[i] == ']':
                    depth -= 1
                i += 1
            inner = rest[1:i - 1]
            stmt, rem_in = Formula._parse_binary(inner, 0)
            if rem_in != '':
                raise ValueError('Trailing junk inside quantifier brackets')
            return Formula(quantifier, var, stmt), rest[i:]
        if string[0] == '(':
            depth = 1
            j = 1
            while depth > 0:
                if j >= len(string):
                    raise ValueError('Unbalanced parentheses')
                if string[j] == '(':
                    depth += 1
                elif string[j] == ')':
                    depth -= 1
                j += 1
            inner_s = string[1:j - 1]
            inner_formula, rem_in = Formula._parse_binary(inner_s, 0)
            if rem_in != '':
                raise ValueError('Trailing junk inside parentheses')
            return inner_formula, string[j:]
        if 'F' <= string[0] <= 'T':
            rname, rest = _parse_relation_name(string)
            if not rest.startswith('('):
                raise ValueError('Expected ( after relation name')
            rest = rest[1:]
            args = []
            if rest.startswith(')'):
                return Formula(rname, ()), rest[1:]
            while True:
                term, rest = Term._parse_prefix(rest)
                args.append(term)
                if rest.startswith(')'):
                    return Formula(rname, tuple(args)), rest[1:]
                if not rest.startswith(','):
                    raise ValueError('Expected , or ) after relation argument')
                rest = rest[1:]
        t1, rest = Term._parse_prefix(string)
        if not rest.startswith('='):
            raise ValueError('Expected = after term or invalid atomic start')
        t2, rest = Term._parse_prefix(rest[1:])
        return Formula('=', [t1, t2]), rest

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        if is_equality(self.root):
            return str(self.arguments[0]) + '=' + str(self.arguments[1])
        if is_relation(self.root):
            return self.root + '(' + ','.join(str(t) for t in self.arguments) + ')'
        if is_unary(self.root):
            return self.root + str(self.first)
        if is_binary(self.root):
            return '(' + str(self.first) + self.root + str(self.second) + ')'
        assert is_quantifier(self.root)
        return self.root + self.variable + '[' + str(self.statement) + ']'

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and str(self) == str(other)
        
    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Formula, str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            string: string to parse, which has a prefix that is a valid
                representation of a formula.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a term followed by an equality
            followed by a constant name (e.g., ``'f(y)=c12'``) or by a variable
            name (e.g., ``'f(y)=x12'``), then the parsed prefix will include
            that entire name (and not just a part of it, such as ``'f(y)=x1'``).
        """
        return Formula._parse_prefix_unary(string)

    @staticmethod
    def parse(string: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        formula, r = Formula._parse_prefix(string)
        assert r == '', 'Incomplete parse of formula'
        return formula

    def constants(self) -> Set[str]:
        """Finds all constant names in the current formula.

        Returns:
            A set of all constant names used in the current formula.
        """
        if is_equality(self.root) or is_relation(self.root):
            result: Set[str] = set()
            for t in self.arguments:
                result |= t.constants()
            return result
        if is_unary(self.root):
            return self.first.constants()
        if is_binary(self.root):
            return self.first.constants() | self.second.constants()
        assert is_quantifier(self.root)
        return self.statement.constants()

    def variables(self) -> Set[str]:
        """Finds all variable names in the current formula.

        Returns:
            A set of all variable names used in the current formula.
        """
        if is_equality(self.root) or is_relation(self.root):
            result = set()
            for t in self.arguments:
                result |= t.variables()
            return result
        if is_unary(self.root):
            return self.first.variables()
        if is_binary(self.root):
            return self.first.variables() | self.second.variables()
        assert is_quantifier(self.root)
        return {self.variable} | self.statement.variables()

    def free_variables(self) -> Set[str]:
        """Finds all variable names that are free in the current formula.

        Returns:
            A set of every variable name that is used in the current formula not
            only within a scope of a quantification on that variable name.
        """
        if is_equality(self.root) or is_relation(self.root):
            result = set()
            for t in self.arguments:
                result |= t.variables()
            return result
        if is_unary(self.root):
            return self.first.free_variables()
        if is_binary(self.root):
            return self.first.free_variables() | self.second.free_variables()
        assert is_quantifier(self.root)
        return self.statement.free_variables() - {self.variable}

    def functions(self) -> Set[Tuple[str, int]]:
        """Finds all function names in the current formula, along with their
        arities.

        Returns:
            A set of pairs of function name and arity (number of arguments) for
            all function names used in the current formula.
        """
        if is_equality(self.root) or is_relation(self.root):
            result: Set[Tuple[str, int]] = set()
            for t in self.arguments:
                result |= t.functions()
            return result
        if is_unary(self.root):
            return self.first.functions()
        if is_binary(self.root):
            return self.first.functions() | self.second.functions()
        assert is_quantifier(self.root)
        return self.statement.functions()

    def relations(self) -> Set[Tuple[str, int]]:
        """Finds all relation names in the current formula, along with their
        arities.

        Returns:
            A set of pairs of relation name and arity (number of arguments) for
            all relation names used in the current formula.
        """
        if is_relation(self.root):
            return {(self.root, len(self.arguments))}
        if is_equality(self.root):
            return set()
        if is_unary(self.root):
            return self.first.relations()
        if is_binary(self.root):
            return self.first.relations() | self.second.relations()
        assert is_quantifier(self.root)
        return self.statement.relations()

    def substitute(self, substitution_map: Mapping[str, Term],
                   forbidden_variables: AbstractSet[str] = frozenset()) -> \
            Formula:
        """Substitutes in the current formula, each constant name `construct` or
        free occurrence of variable name `construct` that is a key in
        `substitution_map` with the term
        `substitution_map`\\ ``[``\\ `construct`\\ ``]``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.
            forbidden_variables: variable names not allowed in substitution
                terms.

        Returns:
            The formula resulting from performing all substitutions. Only
            constant name and variable name occurrences originating in the
            current formula are substituted (i.e., those originating in one of
            the specified substitutions are not subjected to additional
            substitutions).

        Raises:
            ForbiddenVariableError: If a term that is used in the requested
                substitution contains a variable name from `forbidden_variables`
                or a variable name occurrence that becomes bound when that term
                is substituted into the current formula.

        Examples:
            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,x)'), 'x': Term.parse('c')}, {'z'})
            Ay[c=plus(d,x)]

            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,z)')}, {'z'})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: z

            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,y)')})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: y
        """
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)
        # Task 9.2

    def propositional_skeleton(self) -> Tuple[PropositionalFormula,
                                              Mapping[str, Formula]]:
        """Computes a propositional skeleton of the current formula.

        Returns:
            A pair. The first element of the pair is a propositional formula
            obtained from the current formula by substituting every (outermost)
            subformula that has a relation name, equality, or quantifier at its
            root with a propositional variable name, consistently such that
            multiple identical such (outermost) subformulas are substituted with
            the same propositional variable name. The propositional variable
            names used for substitution are obtained, from left to right
            (considering their first occurrence), by calling
            `next`\\ ``(``\\ `~logic_utils.fresh_variable_name_generator`\\ ``)``.
            The second element of the pair is a mapping from each propositional
            variable name to the subformula for which it was substituted.

        Examples:
            >>> formula = Formula.parse('((Ax[x=7]&x=7)|(~Q(y)->x=7))')
            >>> formula.propositional_skeleton()
            (((z1&z2)|(~z3->z2)), {'z1': Ax[x=7], 'z2': x=7, 'z3': Q(y)})
            >>> formula.propositional_skeleton()
            (((z4&z5)|(~z6->z5)), {'z4': Ax[x=7], 'z5': x=7, 'z6': Q(y)})
        """
        # Task 9.8

    @staticmethod
    def from_propositional_skeleton(skeleton: PropositionalFormula,
                                    substitution_map: Mapping[str, Formula]) \
            -> Formula:
        """Computes a predicate-logic formula from a propositional skeleton and
        a substitution map.

        Arguments:
            skeleton: propositional skeleton for the formula to compute,
                containing no constants or operators beyond ``'~'``, ``'->'``,
                ``'|'``, and ``'&'``.
            substitution_map: mapping from each propositional variable name of
                the given propositional skeleton to a predicate-logic formula.

        Returns:
            A predicate-logic formula obtained from the given propositional
            skeleton by substituting each propositional variable name with the
            formula mapped to it by the given map.

        Examples:
            >>> Formula.from_propositional_skeleton(
            ...     PropositionalFormula.parse('((z1&z2)|(~z3->z2))'),
            ...     {'z1': Formula.parse('Ax[x=7]'), 'z2': Formula.parse('x=7'),
            ...      'z3': Formula.parse('Q(y)')})
            ((Ax[x=7]&x=7)|(~Q(y)->x=7))

            >>> Formula.from_propositional_skeleton(
            ...     PropositionalFormula.parse('((z9&z2)|(~z3->z2))'),
            ...     {'z2': Formula.parse('x=7'), 'z3': Formula.parse('Q(y)'),
            ...      'z9': Formula.parse('Ax[x=7]')})
            ((Ax[x=7]&x=7)|(~Q(y)->x=7))
        """
        for operator in skeleton.operators():
            assert is_unary(operator) or is_binary(operator)
        for variable in skeleton.variables():
            assert variable in substitution_map
        # Task 9.10
