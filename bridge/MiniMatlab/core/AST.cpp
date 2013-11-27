#include "AST.hpp"

// global variables (yikes) used to alternate between operations: +, -, *, /
const int PLUS = 0;
const int MINUS = 1;
const int MULT = 2;
const int DIV = 3;

// computes the number of elements in a multi_array<double> object
int len(multi_array<double> *m) {
	int res, i;

	res = 0;
	for (i = 0; i < (int)m->getRank(); ++i) {
		res += m->shape(i);
	}

	return res;
}

// execute all statements
void AST::evaluate() {
	int i;

	for (i = 0; i < (int)stmt_list.size(); ++i) {
		stmt_list[i]->evaluate();
	}
}

// print matrices using formatting similar to Matlab
void Stmt::printExp() {
	int cnt;
	multi_array<double>::iterator it  = out->begin();
  	multi_array<double>::iterator end = out->end();
	// make numbers somewhat uniform in length
	cout.setf(std::ios::fixed);
  	cout.precision(4);

	cnt = 0;
	cout << "    ";
	for(; it != end; it++) {
		cout << *it << "   ";
		cnt++;
		if (out->getRank() > 1 && out->shape(1) == cnt) {
			cout << "\n    ";
			cnt = 0;
		}
	}
	cout << "\n";
}

// evaluate the expression and print it
void PrintStmt::evaluate() {
	out = exp->evaluate();

	cout << "ans =\n";
	printExp();
	// de-allocate if necessary
	if (out->getTemp()) delete out;
}

// bind a multi_array object to a variable in "vars"
void VarStmt::evaluate() {
	out = exp->evaluate();
	multi_array<double> *res; 

	if (is_slice) { // support Matlab way of handling views

		if (out->getRank() == 2) { // 2 dimensions	
			res = new multi_array<double>(out->shape(0), out->shape(1));
			*res = 0; // we must initialize it
			(*res)(*out); // copy elements

		} else if (out->getRank() == 1) { // one dimension
			res = new multi_array<double>(out->shape(0));
			*res = 0; // we must initialize it
			(*res)(*out); // copy elements

		} else {
			cout << "Matrix view assigned to " << name << " has too many dimensions!\n";
			exit(0);
		}

	} else {
		res = out;
	}

	out = res;
	out->setTemp(false); // we want to be able to use the variable again
	if (vars->find(name) != vars->end()) {
		delete (*vars)[name]; // delete the existing multi_array object
		(*vars)[name] = out;
	} else {
		(*vars)[name] = out;
	}

	if (to_print) {
		cout << name << " =\n";
		printExp();
	}
}

// binds a slice (multi_array object) to a variable in vars
void VarSliceStmt::evaluate() {
	out = exp->evaluate();
	string name = var_sl->get_name();

	if (vars->find(name) != vars->end()) {
		multi_array<double> tmp, tmp2;

		if (var_sl->is_two_dim()) {
			Slice *s_1 = var_sl->slice_one();
			Slice *s_2 = var_sl->slice_two();
			tmp = (*(*vars)[name])[_(s_1->get_start(), s_1->get_end(), s_1->get_it())]
						[_(s_2->get_start(), s_2->get_end(), s_2->get_it())];
		} else {
			Slice *s = var_sl->slice_one();
			tmp = (*(*vars)[name])[_(s->get_start(), s->get_it(), s->get_end())];
		}

		if (len(out) > 1) {
			tmp2 = *out;
			tmp(tmp2); // use update to avoid re-allocation
		} else { // if out only contains one element
			tmp = *(out->begin());
			// de-allocate if necessary
			if (out->getTemp()) delete out;
		}

		out = (*vars)[name]; // in case printing is needed

		if (to_print) {
			cout << name << " =\n";
			printExp();
		}
		
	} else { // if vars doesn't contain a variable called "name"
		cout << var_sl->get_name() << " does not exist\n";
		exit(0);
	}
}

// evaluate a for loop
void LoopStmt::evaluate() {
	double i, a, stmt_size, s, e, inval;
	multi_array<double> *s_, *e_, *in, *val;
	// val holds the value for the variable used by the for loop (i=...)
	val = new multi_array<double>(1); 

	// extract the values from the expressions
	s_ = start->evaluate();
	e_ = end->evaluate();
	in = interval->evaluate();

	s = *(s_->begin());
	e = *(e_->begin());
	inval = *(in->begin());

	if (s_->getTemp()) delete s_;
	if (e_->getTemp()) delete e_;
	if (in->getTemp()) delete in;

	// if a variable already exists with "name" we make sure to de-allocate it
	if (vars->find(name) != vars->end()) {
		delete (*vars)[name];
	}

	stmt_size = body->size();
	if (inval > 0) { // if "end" is bigger than "start"
		for (i = s; i <= e; i += inval) {
			*val = i;
			(*vars)[name] = val; // update variable used by for loop
			for (a = 0; a < stmt_size; ++a) {
				body->evaluate(); // execute statements
			}
		}
	} else if (inval < 0) { // if "start" is bigger than "end"
		for (i = s; i >= e; i += inval) {
			*val = i;
			(*vars)[name] = val; // update variable used by for loop
			for (a = 0; a < stmt_size; ++a) {
				body->evaluate(); // execute statements
			}
		}
	} else { // if interval is 0
		cout << "Error: Can not use 0 as interval in for loop!\n";
		exit(0);
	}
}

// make sure two multi_array objects are of the same shape
bool dim_match(multi_array<double> *l, multi_array<double> *r) {
	int l_rank, r_rank, i;

	l_rank = l->getRank();
	r_rank = r->getRank();
	if (l_rank != r_rank) return false; // if dimension don't agree
	
	for (i = 0; i < l_rank; ++i) {
		// if the sizes of each dimension don't agree
		if (l->shape(i) != r->shape(i)) return false;
	}

	return true;
}

// helper method used to reduce duplicate code
multi_array<double> *compute_op(Exp *l, Exp *r, int op) {
	int size_l, size_r;
	multi_array<double> *f, *s, *res;

	// extract the multi_array objects from the two expression
	f = l->evaluate();
	s = r->evaluate();

	size_l = len(f);
	size_r = len(s);

	if (size_l == size_r) { // if the length of the matrices match
		if (!dim_match(f, s)) { // if matrix dimensions don't agree
			cout << "Matrix dimensions are different\n";
			exit(0);
		}
		switch (op) {
			case PLUS: return &(*f + (*s));
			break;
			case MINUS: return &(*f - (*s));
			break;
			case MULT: return &(*f * (*s));
			break;
			case DIV: return &(*f / (*s));
			break;
			default: return NULL;
			break;
		}

	} else if (size_r == 1) {
		switch (op) {
			case PLUS: res = &(*f + (*(s->begin())));
			break;
			case MINUS: res = &(*f - (*(s->begin())));
			break;
			case MULT: res = &(*f * (*(s->begin())));
			break;
			case DIV: res = &(*f / (*(s->begin())));
			break;
			default: res = NULL;
			break;
		}

		if (s->getTemp()) delete s; 

		return res;

	} else if (size_l == 1) {
		switch (op) {
			case PLUS: res = &(*(f->begin()) + (*s));
			break;
			case MINUS: res = &(*(f->begin()) - (*s));
			break;
			case MULT: res = &(*(f->begin()) * (*s));
			break;
			case DIV: res = &(*(f->begin()) / (*s));
			break;
			default: res = NULL;
			break;
		}

		if (f->getTemp()) delete f;

		return res;

	} else {
		cout << "Matrix dimensions are different\n";
		exit(0);
	}
}

// exp + exp
multi_array<double>* PlusExp::evaluate() {
	return compute_op(l, r, PLUS);
}

// exp - exp
multi_array<double>* MinusExp::evaluate() {
	return compute_op(l, r, MINUS);
}

// exp * exp
multi_array<double>* MultExp::evaluate() {
	return compute_op(l, r, MULT);
}

// exp / exp
multi_array<double>* DivExp::evaluate() {
	return compute_op(l, r, DIV);
}

// negate multi_array object (not very efficient)
multi_array<double>* UnaryMinusExp::evaluate() {
	multi_array<double> *tmp = exp->evaluate();

	multi_array<double>::iterator it  = tmp->begin();
  	multi_array<double>::iterator end = tmp->end();

	for(; it != end; it++) {
		*it = -(*it);
	}
	tmp->setTemp(true);

	return tmp;
}

// insert number into multi_array object
multi_array<double>* Number::evaluate() {
	multi_array<double> *res = new multi_array<double>(1);
	*res = num;
	res->setTemp(true);
	return res;
}

// construct random multi_array matrix
multi_array<double>* Random::evaluate() {
	multi_array<double> *res = 
		new multi_array<double>(index->get_left(), index->get_right());
	*res = random<double>(index->get_left(), index->get_right());
	res->setTemp(true);

	return res;
}

// construct multi_array object containing zeros
multi_array<double>* Zeros::evaluate() {
	multi_array<double> *res = 
		new multi_array<double>(index->get_left(), index->get_right());
	*res = 0;
	res->setTemp(true);

	return res;
}

// construct multi_array object containing 1
multi_array<double>* Ones::evaluate() {
	multi_array<double> *res = 
		new multi_array<double>(index->get_left(), index->get_right());
	*res = 1;
	res->setTemp(true);

	return res;
}

// transfer elements from the vector tmp to a multi_array object
multi_array<double>* ArrayExp::evaluate() {
	int i, size;

	size = tmp.size();

	multi_array<double> *res = new multi_array<double>(size);
	*res = 0; // necessary to initiate the object
	
	multi_array<double>::iterator it  = res->begin();
  	multi_array<double>::iterator end = res->end();

	i = 0;
	for(; it != end; it++) {
		multi_array<double> *t = tmp[i]->evaluate();
		*it = *(t->begin());
		// de-allocation is needed
		if (t->getTemp()) delete t; 
		i++;
	}
	res->setTemp(true);

	return res;
}

// fill tmp
void ArrayExp::add(Exp *e) {
	tmp.push_back(e);
}

// return matrix associated with variable
multi_array<double>* Variable::evaluate() {
	if (vars->find(name) != vars->end()) {
		return (*vars)[name];
	} else { // if variable doesn't exist
		cout << name << " has not been initialized!\n"; 
		exit(0);
	}
}

// return matrix slice 
multi_array<double>* VarSl::evaluate() {
	if (vars->find(get_name()) != vars->end()) {
		if (is_2dim) {
			multi_array<double> *r = 
				&((*(*vars)[name])[_(sl1->get_start(), sl1->get_end(), sl1->get_it())]
					[_(sl2->get_start(), sl2->get_end(), sl2->get_it())]).view();
			r->setTemp(true);
			return r;
		} else {
			multi_array<double> *r = 
				&((*(*vars)[name])[_(sl1->get_start(), sl1->get_end(), 
					sl1->get_it())]).view();
			r->setTemp(true);
			return r;
		}
	} else {
		cout << name << " has not been initialized!\n"; 
		exit(0);
	}
}
