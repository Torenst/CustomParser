# CustomParser
Parses JavaScript AST into custom internal representation and interpreter 

This was made to parse and interpret javascript ast on the standard input. It runs some custom grammar that we were given.
 We also had to try and not follow mutations in the code and do it functionally. This is one of the final iterations of
 the project we did, where we implemented mutable variables.

Used Python version 3.11.5 and [acorn ast](https://github.com/acornjs/acorn).
what I used to run my program on linux by piping in javascript text into acorn to get an ast and piping it into my program on standard input.
>cat "PATH\TO\JS\INPUT" | node acorn --ecma2024 | python3 customastparser.py 
