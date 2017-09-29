#!/usr/bin/env python

class ConstraintLanguage:
    def __init__(self, str_=None):
        self.constraint_str = str_
        self.constraint = {}
        self.validConstraints = ['SUBSET', 'MUTEX', 'LOGEQ', 'XOR']
        self.validMemberOperations = ['and', 'or', 'not']
        if str_ is not None:
            self.parse(str_)

    def parse(self, string):
        self.constraint_str = string
        self.parseConstraint()
        self.parseMembers()
        return self.constraint
        print(self.constraint)

    def parseConstraint(self):
        colon_count = self.constraint_str.count(":")
        if colon_count != 2:
            #improper use of constraint indicator
            raise ValueError("Improper use of constraint indicator detected")
        start_index = self.constraint_str.find(":")
        end_index = self.constraint_str.find(":", start_index+1)
        op = self.constraint_str[start_index+1:end_index]
        if op not in self.validConstraints:
            raise ValueError("Provided constraint " + op + "is invalid")
        self.constraint['CONSTRAINT'] = op

    def parseMembers(self):
        start_index = self.constraint_str.find(":")
        end_index = self.constraint_str.find(":", start_index+1)

        left_member = self.constraint_str[0:start_index]
        right_member = self.constraint_str[end_index+1:len(self.constraint_str)]

        self.constraint['LEFT_MEMBER'] = left_member
        self.constraint['RIGHT_MEMBER'] = right_member

    def parseCompoundMembers(self):
        #start with simple case a.and.b
        pass

def main():
    string = "a:SUBSET:b"
    cl = ConstraintLanguage(string)

if __name__ == "__main__":
    main()
