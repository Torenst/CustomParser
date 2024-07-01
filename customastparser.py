import json
import sys

class Result:
    def __init__(self, value):
        self.value = value

    def to_s_expr(self):
        if isinstance(self.value, bool):
            return f"(value (boolean {str(self.value).lower()}))"
        elif isinstance(self.value, int):
            return f"(value (number {self.value}))"
        elif isinstance(self.value, FunctionValue):
            return "(value (function))"
        elif(self.value == None):
            return "(value (void))"
        else:
            # if it is unknown we will just return unknown value so we can fix it later instead of it just returning none.
            return f"(value (unknown: {self.value}))"

class ErrorResult:
    def __init__(self, message):
        self.message = message

    def to_s_expr(self):
        return f'(error "{self.message}")'
    
#adding a heap and a environment
class Heap:
    def __init__(self):
        self.memory = {}
        self.next_address = 0
    
    def __len__(self):
        return len(self.memory)

    def allocate(self, value):
        address = self.next_address
        self.memory[address] = value
        self.next_address += 1
        return address

    def get(self, address):
        return self.memory.get(address)

class Environment:
    def __init__(self):
        self.variables = {}

    def set(self, identifier, address):
        self.variables[identifier] = address

    def get(self, identifier):
        return self.variables.get(identifier)
    
    def copy(self):
        new_env = Environment()
        new_env.variables = self.variables.copy()
        return new_env

class TopLevelTerm:
    def __init__(self, variable_declarations, statements):
        self.variable_declarations = variable_declarations
        self.statements = statements

    def interpret(self, env, heap):
        pass

class Program(TopLevelTerm):
    def __init__(self, variable_declarations, statements):
        super().__init__(variable_declarations, statements)
        self.variable_declarations = variable_declarations
        self.statements = statements

    # Recursively interpret the remaining statements
    def interpret_statements(self, statements, env, heap):
        if not statements:
            return None

        # Interpret the first statement
        statement_result = statements[0].interpret(env, heap)

        if isinstance(statement_result, ErrorResult):
            return statement_result
        # If this is the last statement and it's an AssignmentExpression, return (value (void))
        if len(statements) == 1 and isinstance(statements[0].expression, AssignmentExpression):
            return Result(None)
        
        final_result = self.interpret_statements(statements[1:], env, heap)

        if isinstance(final_result, Identifier):
            final_result = heap.get(env.get(final_result.name))

        if final_result is not None:
            return final_result
    
        # Check if the last statement is an AssignmentExpression, UnaryExpression, or BinaryExpression, or CallExpression
        if (len(statements) == 1 or final_result is None):
            if isinstance(statements[0].expression, (AssignmentExpression, UnaryExpression, BinaryExpression)):                
                return Result(None)
            
        return statement_result

    def interpret(self, env, heap):
        # Interpret variable declarations
        for var_declaration in self.variable_declarations:
            var_declaration.interpret(env, heap)

        # start recursively parsing our statements
        return self.interpret_statements(self.statements, env, heap)
    
class Statement:
    def __init__(self, expression_statement):
        self.expression_statement = expression_statement

    def unparse(self):
        return self.expression.unparse()

    def interpret(self, env, heap):
        return self.expression_statement.interpret(env, heap)

class ExpressionStatement:
    def __init__(self, expression):
        self.expression = expression

    def unparse(self):
        return self.expression.unparse()
    
    def interpret(self, env, heap):
        return self.expression.interpret(env, heap)

class Expression:
    def __init__(self, expression):
        self.expression = expression
    
    def unparse(self):
        pass

    def interpret(self, env, heap):
        pass

class Literal(Expression):
    def __init__(self, value):
        self.value = value
    
    def unparse(self):
        if isinstance(self.value, bool):
            return f"(boolean {str(self.value).lower()})"
        return f"(number {self.value})"

    def interpret(self, env, heap):
        return Result(self.value)

class UnaryExpression(Expression):
    def __init__(self, operator, expr):
        self.operator = operator
        self.expr = expr

    def unparse(self):
        return f"(unary {self.operator} {self.expr.unparse()})"

    def interpret(self, env, heap):
        if isinstance(self.expr, AssignmentExpression):
            return ErrorResult("Void value cannot be used in a unary operation banana")

        result = self.expr.interpret(env, heap)
        if isinstance(result, ErrorResult):
            return result

        if self.operator == '+':
            return result
        elif self.operator == '-':
            if isinstance(result.value, int):
                return Result(-result.value)
            else:
                return ErrorResult("Unary minus applied to non-integer value banana")
        elif self.operator == '!':
            if isinstance(result.value, bool) or type(result.value) is bool:
                return Result(not result.value)
            else:
                return ErrorResult("Logical NOT applied to non-boolean value banana")
        else:
            return ErrorResult(f"Unsupported unary operator: {self.operator} banana")

class BinaryExpression(Expression):
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right

    def unparse(self):
        if self.operator == '==':
            return f"(relational == {self.left.unparse()} {self.right.unparse()})"
        elif self.operator == '<':
            return f"(relational < {self.left.unparse()} {self.right.unparse()})"
        else:
            return f"(arithmetic {self.operator} {self.left.unparse()} {self.right.unparse()})"

    def interpret(self, env, heap):
        if isinstance(self.left, AssignmentExpression):
            return ErrorResult("Void value cannot be used in a binary operation banana")
        
        left_result = self.left.interpret(env, heap)
        right_result = self.right.interpret(env, heap)

        if isinstance(left_result, ErrorResult):
            return left_result
        if isinstance(right_result, ErrorResult):
            return right_result
        
        if (left_result.value == None):
            return left_result
        if (right_result.value == None):
            return right_result
        
        # this should catch anything that is not a int for the + - * / == < operators
        if (type(left_result.value) is not int or type(right_result.value) is not int):
            return ErrorResult("Integer operation applied to non-integer value banana")

        if self.operator in ('+', '-', '*', '/'):
            # error checking to make sure that it is not a boolean since python a bool is a extension of int.
            # might not be needed, but I left it just in case.
            if  isinstance(left_result.value, bool) or isinstance(right_result.value, bool):
                return ErrorResult("Arithmetic operation applied to non-integer value banana")
            

            if not isinstance(left_result.value, int) or not isinstance(right_result.value, int):
                return ErrorResult("Arithmetic operation applied to non-integer value banana")

            if self.operator == '+':
                return Result(left_result.value + right_result.value)
            elif self.operator == '-':
                return Result(left_result.value - right_result.value)
            elif self.operator == '*':
                return Result(left_result.value * right_result.value)
            elif self.operator == '/':
                if right_result.value == 0:
                    return ErrorResult("Division by zero banana")
                return Result(left_result.value // right_result.value)
        elif self.operator == '==':
            if type(left_result.value) != type(right_result.value):
                return ErrorResult("Type comparison applied between different types banana")
            return Result(left_result.value == right_result.value)
        elif self.operator == '<':
            if type(left_result.value) != int or type(right_result.value) != int:
                return ErrorResult("Type comparison applied between different types banana")
            return Result(left_result.value < right_result.value)
        else:
            return ErrorResult(f"Unsupported binary operator: {self.operator} banana")

class LogicalExpression(Expression):
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right

    def unparse(self):
        return f"(logical {self.operator} {self.left.unparse()} {self.right.unparse()})"

    def interpret(self, env, heap):
        left_result = self.left.interpret(env, heap)
        if isinstance(left_result, ErrorResult):
            return left_result
        
        if(type(left_result.value) is not bool):
            return ErrorResult("Logical expression with non-boolean value banana")
        
        if self.operator == '&&':
            # checking it is only booleans.
            if (type(left_result.value) is not bool) or (not isinstance(left_result.value, bool)):
                return ErrorResult("Logical AND applied to non-boolean value banana")

            # short-circuit if left is False
            if not left_result.value:
                return Result(False)
            else:
                return self.right.interpret(env, heap)
        elif self.operator == '||':
            # checking for only booleans
            if  (type(left_result.value) is not bool) or (not isinstance(left_result.value, bool)):
                return ErrorResult("Logical OR applied to non-boolean value banana")
            
            # short-circuit if left is True
            if left_result.value:
                return Result(True)
            else:
                #check the right value to make sure it is right.
                right_result = self.right.interpret(env, heap)
                if isinstance(right_result, ErrorResult):
                    return right_result
                
                if type(right_result.value) is not bool:
                    return ErrorResult("Logical operator applied to non-boolean value banana")
                
                return self.right.interpret(env, heap)
        else:
            return ErrorResult(f"Unsupported logical operator: {self.operator} banana")

class ConditionalExpression(Expression):
    def __init__(self, test, consequent, alternate):
        self.test = test
        self.consequent = consequent
        self.alternate = alternate

    def unparse(self):
        return f"(conditional {self.test.unparse()} {self.consequent.unparse()} {self.alternate.unparse()})"    

    def interpret(self, env, heap):
        if isinstance(self.test, AssignmentExpression):
            return ErrorResult("Void value cannot be used as test in conditional operation banana")
        
        test_result = self.test.interpret(env, heap)
        if isinstance(test_result, ErrorResult):
            return test_result
        
        if (type(test_result.value) is not bool) or (not isinstance(test_result.value, bool)):
                return ErrorResult("Conditional expression with non-boolean test condition banana")

        if test_result.value:
            return self.consequent.interpret(env, heap)
        else:
            return self.alternate.interpret(env, heap)

class FunctionExpression(Expression):
    def __init__(self, parameter, block_statement):
        self.parameter = parameter
        self.block_statement = block_statement

    def unparse(self):
        return f"(function ({self.identifier}) {self.block_statement.unparse()})"    

    def interpret(self, env, heap):
        return Result(FunctionValue(self.parameter.name, self.block_statement, env))

class CallExpression(Expression):
    def __init__(self, expression, arg_expr):
        self.expression = expression
        self.arg_expr = arg_expr
    
    def unparse(self):
        return f"({self.expression.unparse()} ({self.arg_expr.unparse()}))"
    
    def interpret(self, env, heap):
        func_value = self.expression.interpret(env, heap)
        if isinstance(func_value, ErrorResult):
            return func_value
        
        if not isinstance(func_value.value, FunctionValue):
            return ErrorResult("Attempted to call a non-function banana")

        arg_value = self.arg_expr.interpret(env, heap)

        if isinstance(arg_value, ErrorResult):
            return arg_value
        
        return func_value.value.interpret(arg_value, env, heap)
    
class BlockStatement(Expression):
    def __init__(self, variable_declaration, statements, return_statement):
        self.variable_declaration = variable_declaration
        self.statements = statements
        self.return_statement = return_statement

    def unparse(self):
        return f"{{ {' '.join([var_decl.unparse() for var_decl in self.variable_declarations])} {self.return_statement.unparse()} }}"

    def interpret(self, env, heap):
        # we will just leave this as the env passed in instead of trying to copy it since the FunctionValue controls if it is a copy or not.
        block_env = env

        if self.variable_declaration:
            # Interpret variable declarations
            for var_declaration in self.variable_declaration:
                result = var_declaration.interpret(block_env, heap)

                if isinstance(result, ErrorResult):
                    return result
                
                elif isinstance(result, Identifier):
                    result = heap.get(env.get(result.name))

                if result is not None:
                    return result

        # Interpret each of the statements
        for statement in self.statements:
            if isinstance(statement, ErrorResult):
                return statement

            statement_result = statement.interpret(block_env, heap)

            if isinstance(statement_result, ErrorResult):
                return statement_result

            if isinstance(statement_result, Result):
                return statement_result

        # Interpret return statement if present
        if self.return_statement:
            return self.return_statement.interpret(block_env, heap)

        # No return statement
        return Result(None)

class AssignmentExpression(Expression):
    def __init__(self, identifier, expression):
        self.identifier = identifier
        self.expression = expression
    
    def unparse(self):
        return f"({self.identifier} = {self.expression.unparse()})"
    
    def interpret(self, env, heap):
        result = self.expression.interpret(env, heap)
        if isinstance(result, ErrorResult):
            return result

        # Retrieve the address of the variable from the environment
        address = env.get(self.identifier.name)
        if address is not None:
            # Update the variable's value in the heap
            env.set(self.identifier.name, heap.allocate(result))

            if result.value is None:
                return Result(None)
            else:
                return result
        else:
            return ErrorResult(f"Unbound identifier: {self.identifier.name} banana")

class ReturnStatement(Expression):
    def __init__(self, expression):
        self.expression = expression

    def unparse(self):
        return f"(return {self.expression.unparse()})"

    def interpret(self, env, heap):
        if isinstance(self.expression, ErrorResult):
            return self.expression

        return self.expression.interpret(env, heap)

class FunctionValue:
    def __init__(self, parameter, block_statement, env):
        self.parameter = parameter
        self.block_statement = block_statement
        self.env = env

    def interpret(self, arg_value, caller_env, heap):
        
        if self.block_statement.return_statement is None or isinstance(self.block_statement.return_statement.expression, FunctionExpression):
            if self.block_statement.return_statement is None:
                # if it is nested block statements from functions being nested just use the same environment from the top
                # and if it doesn't have a return statement then it is still part of the same env
                call_env = self.env
            elif isinstance(self.block_statement.return_statement.expression.block_statement, BlockStatement):
                # Create a new environment for the function call and add our new parameter  
                call_env = self.env.copy()
            else:
                #if it is none, and we don't have a block statement coming up then it is still the same env
                call_env = self.env
        else:
            call_env = self.env


        call_env.set(self.parameter, arg_value)

        if isinstance(self.block_statement, ErrorResult):
            return self.block_statement

        # Interpret the function body with the new environment (this is calling the function.)
        return self.block_statement.interpret(call_env, heap)

class VariableDeclarator:
    def __init__(self, identifier, expression):
        self.identifier = identifier
        self.expression = expression

    def interpret(self, env, heap):
        pass

class VariableDeclaration:
    def __init__(self, identifier, expression):
        self.identifier = identifier
        self.expression = expression

    def interpret(self, env, heap):
        # make sure the value we returned was not an error.
        if isinstance(self.expression, ErrorResult):
            return self.expression
        
        # Store pointer to new value in environment, or to variable identifier if it is an identifier
        if isinstance(self.expression, Identifier):
            env.set(self.identifier.name, heap.allocate(self.expression))
        else:
            env.set(self.identifier.name, heap.allocate(self.expression.interpret(env, heap)))

class Identifier:
    def __init__(self, name):
        self.name = name

    def interpret(self, env, heap):
        pointer = env.get(self.name)
        if pointer is not None:
            if isinstance(pointer, int):
                value = heap.get(pointer)
                if value is not None:
                    return value
                else:
                    return ErrorResult(f"Invalid pointer: {pointer} banana")
            else:
                return pointer
        else:
            return ErrorResult(f"Variable '{self.name}' is not defined banana")


def parse_variable_declaration(node):
    declarators = [parse_variable_declarator(decl) for decl in node['declarations']]
    return [VariableDeclaration(decl.identifier, decl.expression) for decl in declarators]

def parse_variable_declarator(node):
    identifier = Identifier(node['id']['name'])
    expression = parse_expression(node['init'])
    return VariableDeclarator(identifier, expression)

# starts by parsing the program, then parses the expression statement then parsing the expressions which is recursive.
def parse_program(json_ast):
    variable_declarations = []
    statements = []

    for node in json_ast["body"]:
        if node["type"] == "VariableDeclaration":
            variable_declarations.extend(parse_variable_declaration(node))
        else:
            statements.append(parse_statement(node))

    return Program(variable_declarations, statements)

def parse_statement(node):
    if isinstance(node, list):
        return [parse_statement(stmt) for stmt in node]
    elif node["type"] == "VariableDeclaration":
        return parse_variable_declaration(node)
    elif node["type"] == "ExpressionStatement":
        return parse_expression_statement(node)
    else:
        return ErrorResult("Invalid node type for statement: {} banana".format(node["type"]))

def parse_expression_statement(node):
    if node["type"] == "ExpressionStatement":
        return ExpressionStatement(parse_expression(node["expression"]))
    else:
        return ErrorResult("Invalid type for expression statement: {} banana".format(node["type"]))

def parse_expression(node):
    if node["type"] == "Literal":
        # removed the check for bool/int here
        return Literal(node["value"])
    elif node["type"] == "UnaryExpression":
        return UnaryExpression(node["operator"], parse_expression(node["argument"]))
    elif node["type"] == "BinaryExpression":
        return BinaryExpression(
            node["operator"],
            parse_expression(node["left"]),
            parse_expression(node["right"])
        )
    elif node["type"] == "LogicalExpression":
        return LogicalExpression(
            node["operator"],
            parse_expression(node["left"]),
            parse_expression(node["right"])
        )
    elif node["type"] == "ConditionalExpression":
        return ConditionalExpression(
            parse_expression(node["test"]),
            parse_expression(node["consequent"]),
            parse_expression(node["alternate"])
        )
    elif node["type"] == "Identifier":
        return Identifier(node["name"])
    elif node["type"] == "FunctionExpression":
        return parse_function_expression(node)
    elif node["type"] == "CallExpression":
        return parse_call_expression(node)
    elif node["type"] == "AssignmentExpression":
        return parse_assignment_expression(node)

def parse_assignment_expression(node):
    identifier = parse_expression(node["left"])
    expression = parse_expression(node["right"])
    return AssignmentExpression(identifier, expression)

def parse_function_expression(node):
    parameter = Identifier(node["params"][0]["name"])
    body = parse_block_statement(node["body"])
    return FunctionExpression(parameter, body)

def parse_call_expression(node):
    function_expression = parse_expression(node["callee"])
    # we only have 1 parameter so only one argument
    argument_expression = parse_expression(node["arguments"][0])
    return CallExpression(function_expression, argument_expression)

def parse_block_statement(node):
    variable_declaration = None
    statements = []
    return_statement = None

    # parse each statement that is inside the block
    for statement_node in node["body"]:
        if statement_node["type"] == "VariableDeclaration":
            if variable_declaration is not None:
                return ErrorResult("Multiple variable declarations found in block statement banana")
            variable_declaration = parse_variable_declaration(statement_node)
        elif statement_node["type"] == "ReturnStatement":
            return_statement = parse_return_statement(statement_node)
        elif statement_node["type"] == "ExpressionStatement":
            expression_statement = parse_expression_statement(statement_node)
            statements.append(expression_statement)
        else:
            # if there is any other type, it is an invalid type.
            return ErrorResult("Invalid type in block statement: {} banana".format(statement_node["type"]))

    return BlockStatement(variable_declaration, statements, return_statement)

def parse_return_statement(node):
    expression = parse_expression(node["argument"])
    return ReturnStatement(expression)


if __name__ == "__main__":
    # Read AST as a string from standard input
    ast_str = sys.stdin.read()

    try:
        ast = json.loads(ast_str)

        parsed_program = parse_program(ast)

        #create our environment and heap to keep track of our values and functions.
        env = Environment()
        heap = Heap()
        result = parsed_program.interpret(env, heap)

        print(result.to_s_expr())

    except json.JSONDecodeError as e:
        print(f"Error parsing AST Json String: {e}")
