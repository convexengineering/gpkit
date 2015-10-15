"""
Takes a .txt file of variable name strings and rewrites
them as a .py file of GPkit variable initializations

$ ls
example.txt
$ python mkvar.py example.txt
$ ls
example.py example.txt

- - - - - - - example.txt - - - - - - -
foo
bar
- - - - - - - - - - - - - - - - - - - - 

                | |
                | |
               \   /
                \ /
                 v

- - - - - - - example.py - - - - - - -
foo = Variable('foo')
bar = Variable('bar')
- - - - - - - - - - - - - - - - - - - - 
"""
import sys

def initialise_variable_list(fname):
    with open(fname) as infile:
        with open(fname.replace('.txt','.py'), 'w') as outfile:
            for line in infile:
                var = line.rstrip('\n')
                outfile.write(var + " = Variable(\'" + var + "\')\n")

if __name__ == "__main__":
    initialise_variable_list(sys.argv[1])
