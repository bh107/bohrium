#ifndef __AST_H
#define __AST_H

#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include "../../cpp/bh/bh.hpp"
using namespace std;
using namespace bh;

/**
 * Class: Exp
 * 
 * One of two main classes used by the parser, stores all possible expressions.
 * E.g. 2 * [1,2,3], variables, matrices etc.
 */
class Exp {

public:
	Exp() { }

	virtual multi_array<double>* evaluate() = 0; // all sub-classes need this method
};

/**
 * Sub-class: PlusExp
 * 
 * computes l + r
 */
class PlusExp : public Exp {

private:
	Exp *l;
	Exp *r;

public:
	PlusExp(Exp *a, Exp *b) : l(a), r(b) { }

	multi_array<double>* evaluate();
};

/**
 * Sub-class: MinusExp
 * 
 * computes l - r
 */
class MinusExp : public Exp {

private:
	Exp *l;
	Exp *r;

public:
	MinusExp(Exp *a, Exp *b) : l(a), r(b) { }

	multi_array<double>* evaluate();
};

/**
 * Sub-class: MultExp
 * 
 * computes l * r
 */
class MultExp : public Exp {

private:
	Exp *l;
	Exp *r;

public:
	MultExp(Exp *a, Exp *b) : l(a), r(b) { }

	multi_array<double>* evaluate();
};

/**
 * Sub-class: DivExp
 * 
 * computes l / r
 */
class DivExp : public Exp {

private:
	Exp *l;
	Exp *r;

public:
	DivExp(Exp *a, Exp *b) : l(a), r(b) { }

	multi_array<double>* evaluate();
};

/**
 * Sub-class: UnaryMinusExp
 * 
 * negates the given expression, i.e. arrays, matrices and numbers
 */
class UnaryMinusExp : public Exp {

private:
	Exp *exp;

public:
	UnaryMinusExp(Exp *e) : exp(e) { }

	multi_array<double>* evaluate();
};

/**
 * Sub-class: Number
 * 
 * puts numbers into multi_array<double> of size 1
 */
class Number : public Exp {

private:
	double num;
public:
	Number(double n) : num(n) { }

	multi_array<double>* evaluate();
};

/**
 * Sub-class: ArrayExp
 * 
 * puts [1,2,3] into multi_array<double>(1, 3)
 */
class ArrayExp : public Exp {

private:
	vector<Exp *> tmp;
public:
	ArrayExp() { }

	multi_array<double>* evaluate();

	void add(Exp *e);
};

/**
 * Class: Slices
 * 
 * stores information for slices: A(1:2:10) => start = 1, it = 2 and end = 10
 */
class Slice {

private:
	Exp *start;
	Exp *it;
	Exp *end;

public:
	Slice(Exp *s, Exp *i, Exp *e) : start(s), it(i), end(e) { }

	/**
 	* Getter method
 	* 
 	* returns start as a double
 	*/
	double get_start() {
		multi_array<double> *out = start->evaluate();
		double s = *out->begin();
		// we remember to de-allocate out if need be
		if (out->getTemp()) delete out; 

		return s - 1;
	}

	/**
 	* Getter method
 	* 
 	* returns it as a double
 	*/
	double get_it() {
		multi_array<double> *out = it->evaluate();
		double i = *out->begin();
		// we remember to de-allocate out if need be
		if (out->getTemp()) delete out;

		return i;
	}

	/**
 	* Getter method
 	* 
 	* returns end as a double
 	*/
	double get_end() {
		multi_array<double> *out = end->evaluate();
		double e = *out->begin();
		// we remember to de-allocate out if need be
		if (out->getTemp()) delete out;

		return e;
	}
};

/**
 * Class: Index
 * 
 * stores information for matrix indexes, A(1, i + 5) => left = 1 and right = i + 5
 */
class Index {

private:
	Exp *left;
	Exp *right;
	bool is_both; // if two dimensional

public:
	Index(Exp *l, Exp *r) : left(l), right(r), is_both(true) { }
	
	Index(Exp *e) : left(e), right(e), is_both(false) { }

	/**
 	* Getter method
 	* 
 	* returns left as a double
 	*/
	double get_left() {
		multi_array<double> *out = left->evaluate();
		double l = *out->begin();
		// we remember to de-allocate out if need be
		if (out->getTemp() && is_both) delete out;

		return l;
	}

	/**
 	* Getter method
 	* 
 	* returns right as a double
 	*/
	double get_right() {
		multi_array<double> *out = right->evaluate();
		double r = *out->begin();
		// we remember to de-allocate out if need be
		if (out->getTemp()) delete out;

		return r;
	}
};

/**
 * Sub-class: Random
 * 
 * creates a random matrix 
 */
class Random : public Exp {

private:
	Index *index;
public:
	Random(Index *i) : index(i) { }

	multi_array<double>* evaluate();
};

/**
 * Sub-class: Zeros
 * 
 * creates a matrix with all elements set to zero
 */
class Zeros : public Exp {

private:
	Index *index;
public:
	Zeros(Index *i) : index(i) { }

	multi_array<double>* evaluate();
};

/**
 * Sub-class: Ones
 * 
 * creates a matrix with all elements set to one
 */
class Ones : public Exp {

private:
	Index *index;
public:
	Ones(Index *i) : index(i) { }

	multi_array<double>* evaluate();
};


/**
 * Sub-class: Variable
 * 
 * fetches a multi_array<double> object in the variable table using name
 */
class Variable : public Exp {

protected:
	string name;
	map<string, multi_array<double>* > *vars;

public:
	Variable(string n, map<string, multi_array<double>* > *v) : name(n), vars(v) { }

	multi_array<double>* evaluate();

	string get_name() {
		return name;
	}
};

/**
 * Sub-class: VarSl
 * 
 * creates a ''sliced view'' of an existing variable 
 */
class VarSl : public Variable {

private:
	Slice *sl1;
	Slice *sl2;
	bool is_2dim;

public:
	VarSl(string n, Slice *s, map<string, multi_array<double>* > *v) : 
		Variable(n, v), sl1(s), is_2dim(false) { }

	VarSl(string n, Slice *s1, Slice *s2, map<string, multi_array<double>* > *v) : 			Variable(n, v), sl1(s1), sl2(s2), is_2dim(true) { }

	multi_array<double>* evaluate();

	bool is_two_dim() {
		return is_2dim;
	}

	Slice* slice_one() {
		return sl1;
	}

	Slice* slice_two() {
		return sl2;
	}

};

/**
 * Class: Stmt
 * 
 * The second of the main classes used by the parser, stores statements.
 * E.g. A = Exp, A(1:4, 1:end) = Exp etc.
 */
class Stmt {

protected:
  	Exp *exp;
	multi_array<double>* out;

public:
	Stmt(Exp *e) : exp(e) { }

	virtual void evaluate() = 0; // all sub-classes need this method

	void printExp();

};

/**
 * Class: AST
 * 
 * abstract syntax tree used by for loops.
 */
class AST {

private:
	vector<Stmt*> stmt_list;

public:
	void evaluate();

	int size() {
		return stmt_list.size();
	}

	void push(Stmt *s) {
		stmt_list.push_back(s);
	}

};

/**
 * Sub-class: PrintStmt
 * 
 * prints multi_array<double> to the screen
 */
class PrintStmt : public Stmt {

public:
  	PrintStmt(Exp *e) : Stmt(e) { }

	void evaluate();

};

/**
 * Sub-class: LoopStmt
 * 
 * executes for loops
 */
class LoopStmt : public Stmt {

private:
	string name;
	Exp *start, *interval, *end;
  	AST *body;
	map<string, multi_array<double>* > *vars;

public:
	LoopStmt(string n, Exp *s, Exp *i, Exp *e, AST *b, map<string, 
		multi_array<double>* > *v) : Stmt(NULL), name(n), 
			start(s), interval(i), end(e), body(b),  vars(v) { }

	void evaluate();

};

/**
 * Sub-class: VarStmt
 * 
 * binds variables to a multi_array<double> object
 */
class VarStmt : public Stmt {

private:
  	string name;
	map<string, multi_array<double>* > *vars;
	bool to_print;
	bool is_slice;

public:
  	VarStmt(string n, Exp *e, bool t_p, bool i_s, 
		map<string, multi_array<double>* > *v) : 
		Stmt(e), name(n), vars(v), to_print(t_p), is_slice(i_s) { }

	void evaluate();

};

/**
 * Sub-class: VarSliceStmt
 * 
 * updates slice elements
 */
class VarSliceStmt : public Stmt {

private:
  	VarSl *var_sl;
	map<string, multi_array<double>* > *vars;
	bool to_print;

public:
  	VarSliceStmt(VarSl *v_s, Exp *e, bool t_p, 
		map<string, multi_array<double>* > *v) : Stmt(e), var_sl(v_s), 
			vars(v), to_print(t_p) { }

	void evaluate();

};

#endif
