/********************************************************
 * lexer.l                                              *
 * Defines the lexical analyser, which tokenizes the    *
 * input and sends it to the parser: parser.ypp         *
 ********************************************************/
%{
#include "par.tab.hpp"
#include <iostream>
using namespace std;
%}

/* options used to suppress warnings/errors */
%option noyywrap
%option nounput
%option yylineno

/* variables used to store one digit numbers and names (no danish characters) */
DIGIT    [0-9]
ID       [a-zA-Z][a-zA-Z0-9_]*

/* tokens are defined below using regular expressions */
%%

{DIGIT}+ 				{ yylval.num = atof(yytext); return NUM; } 
{DIGIT}+"."{DIGIT}* 	{ char *p; yylval.num = strtod(yytext, &p); return NUM; } 

"for"					{ return FOR; }
"end"					{ return ENDER; }

"rand"					{ return RAND; }
"zeros"					{ return ZEROS; }
"ones"					{ return ONES; }

"exit"					{ return EXIT; }

{ID} 					{ yylval.name = strdup(yytext); return VARIABLE; }

"+"		  				{ return PLUS; }
"-"		  				{ return MINUS; }
"*"   					{ return MULT; }
"/"   					{ return DIVIDE; }
"="   					{ return EQ; }

"("     				{ return LP; }
")"     				{ return RP; }
"["      				{ return LS; }
"]"      				{ return RS; }

","      				{ return COMMA; }
":"      				{ return COLON; }
";"      				{ return STOP; }
"\n"      				{ return NL; }

"%"[^\n]* 				{ /* Ignore one-line comments */ }

[ \t]+ 					{ /* Ignore whitespace */ }

<<EOF>>  				{ return 0; }

.        				{ cerr << "Unrecognized token!\n"; exit(1); }

%%

/* we wish to print errors, line numbers are not always accurate */
void yyerror(char *s) {
	fprintf(stderr, "Error occured near line %d: %s\n", yylineno, s);
	exit(0);
}
