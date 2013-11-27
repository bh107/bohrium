/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison implementation for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2011 Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.5"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Copy the first part of user declarations.  */

/* Line 268 of yacc.c  */
#line 4 "MiniMatlab/parser.ypp"

#include "AST.hpp"
#include <stdio.h>
#include <string>
#include <cmath>
#include <sstream>

extern "C" FILE *yyin;

//-- Lexer prototype required by bison, aka getNextToken()
int yylex(); 
int yyerror(char *p);

map<string, multi_array<double>* > vars;


/* Line 268 of yacc.c  */
#line 88 "MiniMatlab/.tab.cpp"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

/* "%code requires" blocks.  */

/* Line 288 of yacc.c  */
#line 22 "MiniMatlab/parser.ypp"
 #include "AST.hpp" 


/* Line 288 of yacc.c  */
#line 116 "MiniMatlab/.tab.cpp"

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     NUM = 258,
     VARIABLE = 259,
     NL = 260,
     PLUS = 261,
     MINUS = 262,
     MULT = 263,
     DIVIDE = 264,
     LP = 265,
     RP = 266,
     LS = 267,
     RS = 268,
     FOR = 269,
     ENDER = 270,
     EQ = 271,
     COMMA = 272,
     COLON = 273,
     STOP = 274,
     EXPON = 275,
     RAND = 276,
     ZEROS = 277,
     ONES = 278
   };
#endif



#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 293 of yacc.c  */
#line 24 "MiniMatlab/parser.ypp"

  AST *t_ast;
  Stmt *t_stmt;
  Exp *t_exp;
  Variable *t_var;
  NArray *t_array; 
  Slice *t_slice;
  Index *t_index;
  VarSl *t_varSl;
  double num; 
  char *name;



/* Line 293 of yacc.c  */
#line 171 "MiniMatlab/.tab.cpp"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 343 of yacc.c  */
#line 183 "MiniMatlab/.tab.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  33
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   202

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  24
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  22
/* YYNRULES -- Number of rules.  */
#define YYNRULES  69
/* YYNRULES -- Number of states.  */
#define YYNSTATES  132

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   278

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint8 yyprhs[] =
{
       0,     0,     3,     5,     8,    10,    13,    15,    16,    26,
      38,    41,    46,    51,    54,    59,    64,    66,    68,    70,
      74,    78,    82,    84,    88,    92,    94,    97,   100,   102,
     104,   106,   110,   116,   120,   124,   126,   130,   134,   136,
     139,   142,   144,   146,   148,   152,   154,   159,   166,   168,
     172,   176,   180,   182,   186,   190,   192,   195,   198,   200,
     202,   204,   208,   210,   212,   214,   218,   222,   227,   232
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      25,     0,    -1,    29,    -1,    25,    29,    -1,    29,    -1,
      26,    29,    -1,    17,    -1,    -1,    14,     4,    16,    45,
      18,    45,    27,    26,    15,    -1,    14,     4,    16,    45,
      18,    45,    18,    45,    27,    26,    15,    -1,    31,     5,
      -1,     4,    16,    31,     5,    -1,    39,    16,    31,     5,
      -1,    31,    19,    -1,     4,    16,    31,    19,    -1,    39,
      16,    31,    19,    -1,    28,    -1,     5,    -1,    31,    -1,
      30,    17,    31,    -1,    31,     6,    32,    -1,    31,     7,
      32,    -1,    32,    -1,    32,     8,    33,    -1,    32,     9,
      33,    -1,    33,    -1,     6,    45,    -1,     7,    45,    -1,
      45,    -1,    18,    -1,    35,    -1,    35,    18,    35,    -1,
      35,    18,    35,    18,    35,    -1,    35,     6,    36,    -1,
      35,     7,    36,    -1,    36,    -1,    36,     8,    37,    -1,
      36,     9,    37,    -1,    37,    -1,     6,    38,    -1,     7,
      38,    -1,    38,    -1,     3,    -1,     4,    -1,    10,    35,
      11,    -1,    15,    -1,     4,    10,    34,    11,    -1,     4,
      10,    34,    17,    34,    11,    -1,    41,    -1,    41,    17,
      41,    -1,    41,     6,    42,    -1,    41,     7,    42,    -1,
      42,    -1,    42,     8,    43,    -1,    42,     9,    43,    -1,
      43,    -1,     6,    44,    -1,     7,    44,    -1,    44,    -1,
       3,    -1,     4,    -1,    10,    41,    11,    -1,     3,    -1,
       4,    -1,    39,    -1,    12,    30,    13,    -1,    10,    31,
      11,    -1,    21,    10,    40,    11,    -1,    22,    10,    40,
      11,    -1,    23,    10,    40,    11,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint8 yyrline[] =
{
       0,    53,    53,    54,    57,    58,    61,    62,    65,    67,
      71,    72,    73,    74,    75,    76,    77,    78,    82,    83,
      87,    88,    89,    92,    93,    94,    97,    98,    99,   102,
     103,   104,   105,   108,   109,   110,   113,   114,   115,   118,
     119,   120,   123,   124,   125,   126,   129,   130,   133,   134,
     137,   138,   139,   142,   143,   144,   147,   148,   149,   152,
     153,   154,   157,   158,   159,   160,   161,   162,   164,   166
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "NUM", "VARIABLE", "NL", "PLUS", "MINUS",
  "MULT", "DIVIDE", "LP", "RP", "LS", "RS", "FOR", "ENDER", "EQ", "COMMA",
  "COLON", "STOP", "EXPON", "RAND", "ZEROS", "ONES", "$accept", "prog",
  "loop_stmts", "delim", "loop", "stmt", "array_exp", "exp", "term",
  "sfactor", "slice", "exp_slice", "term_slice", "sfactor_slice",
  "factor_slice", "var", "index", "exp_index", "term_index",
  "sfactor_index", "factor_index", "factor", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    24,    25,    25,    26,    26,    27,    27,    28,    28,
      29,    29,    29,    29,    29,    29,    29,    29,    30,    30,
      31,    31,    31,    32,    32,    32,    33,    33,    33,    34,
      34,    34,    34,    35,    35,    35,    36,    36,    36,    37,
      37,    37,    38,    38,    38,    38,    39,    39,    40,    40,
      41,    41,    41,    42,    42,    42,    43,    43,    43,    44,
      44,    44,    45,    45,    45,    45,    45,    45,    45,    45
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     2,     1,     2,     1,     0,     9,    11,
       2,     4,     4,     2,     4,     4,     1,     1,     1,     3,
       3,     3,     1,     3,     3,     1,     2,     2,     1,     1,
       1,     3,     5,     3,     3,     1,     3,     3,     1,     2,
       2,     1,     1,     1,     3,     1,     4,     6,     1,     3,
       3,     3,     1,     3,     3,     1,     2,     2,     1,     1,
       1,     3,     1,     1,     1,     3,     3,     4,     4,     4
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,    62,    63,    17,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    16,     2,     0,    22,    25,    64,    28,
       0,     0,    63,    64,    26,    27,     0,     0,    18,     0,
       0,     0,     0,     1,     3,    10,     0,     0,    13,     0,
       0,     0,    42,    43,     0,     0,     0,    45,    29,     0,
      30,    35,    38,    41,     0,    66,    65,     0,     0,    59,
      60,     0,     0,     0,     0,    48,    52,    55,    58,     0,
       0,    20,    21,    23,    24,     0,    39,    40,     0,    46,
       0,     0,     0,     0,     0,     0,    11,    14,    19,     0,
      56,    57,     0,    67,     0,     0,     0,     0,     0,    68,
      69,    12,    15,    44,     0,    33,    34,    31,    36,    37,
       0,    61,    50,    51,    49,    53,    54,    47,     0,     7,
      32,     6,     0,     0,     7,     0,     4,     0,     8,     5,
       0,     9
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,    12,   125,   123,    13,   126,    27,    15,    16,    17,
      49,    50,    51,    52,    53,    23,    64,    65,    66,    67,
      68,    19
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -59
static const yytype_int16 yypact[] =
{
      88,   -59,    57,   -59,   133,   133,   112,   112,     3,    15,
      19,    32,    14,   -59,   -59,    25,    81,   -59,    -5,   -59,
     143,   112,    42,   -59,   -59,   -59,   168,    -7,   122,    38,
     166,   166,   166,   -59,   -59,   -59,   112,   112,   -59,   112,
     112,   112,   -59,   -59,    12,    12,   156,   -59,   -59,    90,
      27,   130,   -59,   -59,    55,   -59,   -59,   112,   133,   -59,
     -59,   102,   102,   166,    52,    97,   132,   -59,   -59,    67,
      70,    81,    81,   -59,   -59,    77,   -59,   -59,   171,   -59,
     143,   156,   156,   156,   156,   156,   -59,   -59,   122,    79,
     -59,   -59,   174,   -59,   166,   166,   166,   166,   166,   -59,
     -59,   -59,   -59,   -59,    74,   130,   130,    33,   -59,   -59,
     133,   -59,   132,   132,   145,   -59,   -59,   -59,   156,   147,
     161,   -59,   133,    88,    82,    43,   -59,    88,   -59,   -59,
      65,   -59
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
     -59,   -59,   -14,    -3,   -59,     1,   -59,     2,   150,   144,
      37,   -42,   107,   106,   148,     0,   163,   -58,   103,   104,
     138,    -2
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint8 yytable[] =
{
      18,    14,    24,    25,    78,    92,    56,    29,    26,    28,
      57,    41,    18,    34,    33,    42,    43,     1,     2,     3,
       4,     5,    46,    54,     6,    30,     7,    47,     8,    31,
      35,    36,    37,    81,    82,     9,    10,    11,   114,    81,
      82,   107,    32,    75,    38,    83,     1,     2,     3,     4,
       5,   118,    20,     6,    58,     7,    89,     8,   128,    88,
      86,    36,    37,    93,     9,    10,    11,    20,     1,     2,
       3,     4,     5,    21,    87,     6,   120,     7,    99,     8,
     131,   100,   101,    36,    37,   117,     9,    10,    11,    39,
      40,     1,     2,     3,     4,     5,   102,   110,     6,   121,
       7,    79,     8,    94,    95,    59,    60,    80,   119,     9,
      10,    11,    63,   130,    96,     1,    22,   104,     4,     5,
     124,   127,     6,    18,     7,    18,   129,    18,    36,    37,
      18,   129,     0,     9,    10,    11,     1,    22,    84,    85,
      97,    98,     0,     6,     0,     7,    42,    43,     0,    44,
      45,    94,    95,    46,     9,    10,    11,     0,    47,    42,
      43,    48,    44,    45,   121,   122,    46,    81,    82,    59,
      60,    47,    61,    62,    36,    37,    63,    81,    82,    55,
      94,    95,   103,    73,    74,   111,    71,    72,   105,   106,
     108,   109,    76,    77,    69,    70,     0,   112,   113,    90,
      91,   115,   116
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-59))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
       0,     0,     4,     5,    46,    63,    13,     4,     6,     7,
      17,    16,    12,    12,     0,     3,     4,     3,     4,     5,
       6,     7,    10,    21,    10,    10,    12,    15,    14,    10,
       5,     6,     7,     6,     7,    21,    22,    23,    96,     6,
       7,    83,    10,    41,    19,    18,     3,     4,     5,     6,
       7,    18,    10,    10,    16,    12,    58,    14,    15,    57,
       5,     6,     7,    11,    21,    22,    23,    10,     3,     4,
       5,     6,     7,    16,    19,    10,   118,    12,    11,    14,
      15,    11,     5,     6,     7,    11,    21,    22,    23,     8,
       9,     3,     4,     5,     6,     7,    19,    18,    10,    17,
      12,    11,    14,     6,     7,     3,     4,    17,   110,    21,
      22,    23,    10,   127,    17,     3,     4,    80,     6,     7,
     122,   124,    10,   123,    12,   125,   125,   127,     6,     7,
     130,   130,    -1,    21,    22,    23,     3,     4,     8,     9,
       8,     9,    -1,    10,    -1,    12,     3,     4,    -1,     6,
       7,     6,     7,    10,    21,    22,    23,    -1,    15,     3,
       4,    18,     6,     7,    17,    18,    10,     6,     7,     3,
       4,    15,     6,     7,     6,     7,    10,     6,     7,    11,
       6,     7,    11,    39,    40,    11,    36,    37,    81,    82,
      84,    85,    44,    45,    31,    32,    -1,    94,    95,    61,
      62,    97,    98
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     5,     6,     7,    10,    12,    14,    21,
      22,    23,    25,    28,    29,    31,    32,    33,    39,    45,
      10,    16,     4,    39,    45,    45,    31,    30,    31,     4,
      10,    10,    10,     0,    29,     5,     6,     7,    19,     8,
       9,    16,     3,     4,     6,     7,    10,    15,    18,    34,
      35,    36,    37,    38,    31,    11,    13,    17,    16,     3,
       4,     6,     7,    10,    40,    41,    42,    43,    44,    40,
      40,    32,    32,    33,    33,    31,    38,    38,    35,    11,
      17,     6,     7,    18,     8,     9,     5,    19,    31,    45,
      44,    44,    41,    11,     6,     7,    17,     8,     9,    11,
      11,     5,    19,    11,    34,    36,    36,    35,    37,    37,
      18,    11,    42,    42,    41,    43,    43,    11,    18,    45,
      35,    17,    18,    27,    45,    26,    29,    27,    15,    29,
      26,    15
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  However,
   YYFAIL appears to be in use.  Nevertheless, it is formally deprecated
   in Bison 2.4.2's NEWS entry, where a plan to phase it out is
   discussed.  */

#define YYFAIL		goto yyerrlab
#if defined YYFAIL
  /* This is here to suppress warnings from the GCC cpp's
     -Wunused-macros.  Normally we don't worry about that warning, but
     some users do, and we want to make it easy for users to remove
     YYFAIL uses, which will produce warnings from Bison 2.5.  */
#endif

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* This macro is provided for backward compatibility. */

#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (0, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  YYSIZE_T yysize1;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = 0;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - Assume YYFAIL is not used.  It's too flawed to consider.  See
       <http://lists.gnu.org/archive/html/bison-patches/2009-12/msg00024.html>
       for details.  YYERROR is fine as it does not invoke this
       function.
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                yysize1 = yysize + yytnamerr (0, yytname[yyx]);
                if (! (yysize <= yysize1
                       && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                  return 2;
                yysize = yysize1;
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  yysize1 = yysize + yystrlen (yyformat);
  if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
    return 2;
  yysize = yysize1;

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:

/* Line 1806 of yacc.c  */
#line 53 "MiniMatlab/parser.ypp"
    { (yyval.t_ast) = new AST(); if ((yyvsp[(1) - (1)].t_stmt) != NULL) (yyvsp[(1) - (1)].t_stmt)->evaluate(); }
    break;

  case 3:

/* Line 1806 of yacc.c  */
#line 54 "MiniMatlab/parser.ypp"
    { if ((yyvsp[(2) - (2)].t_stmt) != NULL) (yyvsp[(2) - (2)].t_stmt)->evaluate(); }
    break;

  case 4:

/* Line 1806 of yacc.c  */
#line 57 "MiniMatlab/parser.ypp"
    { (yyval.t_ast) = new AST(); if ((yyvsp[(1) - (1)].t_stmt) != NULL) (yyval.t_ast)->stmt_list.push_back((yyvsp[(1) - (1)].t_stmt)); }
    break;

  case 5:

/* Line 1806 of yacc.c  */
#line 58 "MiniMatlab/parser.ypp"
    { if ((yyvsp[(2) - (2)].t_stmt) != NULL) (yyvsp[(1) - (2)].t_ast)->stmt_list.push_back((yyvsp[(2) - (2)].t_stmt)); }
    break;

  case 8:

/* Line 1806 of yacc.c  */
#line 65 "MiniMatlab/parser.ypp"
    { 
	(yyval.t_stmt) = new LoopStmt((yyvsp[(2) - (9)].name), (yyvsp[(4) - (9)].t_exp), new Number(1), (yyvsp[(6) - (9)].t_exp), (yyvsp[(8) - (9)].t_ast), &vars); }
    break;

  case 9:

/* Line 1806 of yacc.c  */
#line 67 "MiniMatlab/parser.ypp"
    { 
	(yyval.t_stmt) = new LoopStmt((yyvsp[(2) - (11)].name), (yyvsp[(4) - (11)].t_exp), (yyvsp[(6) - (11)].t_exp), (yyvsp[(8) - (11)].t_exp), (yyvsp[(10) - (11)].t_ast), &vars); }
    break;

  case 10:

/* Line 1806 of yacc.c  */
#line 71 "MiniMatlab/parser.ypp"
    { (yyval.t_stmt) = new PrintStmt((yyvsp[(1) - (2)].t_exp)); }
    break;

  case 11:

/* Line 1806 of yacc.c  */
#line 72 "MiniMatlab/parser.ypp"
    { (yyval.t_stmt) = new VarStmt((yyvsp[(1) - (4)].name), (yyvsp[(3) - (4)].t_exp), true, &vars); }
    break;

  case 12:

/* Line 1806 of yacc.c  */
#line 73 "MiniMatlab/parser.ypp"
    { (yyval.t_stmt) = new VarSliceStmt((yyvsp[(1) - (4)].t_varSl), (yyvsp[(3) - (4)].t_exp), true, &vars); }
    break;

  case 13:

/* Line 1806 of yacc.c  */
#line 74 "MiniMatlab/parser.ypp"
    { (yyval.t_stmt) = NULL; }
    break;

  case 14:

/* Line 1806 of yacc.c  */
#line 75 "MiniMatlab/parser.ypp"
    { (yyval.t_stmt) = new VarStmt((yyvsp[(1) - (4)].name), (yyvsp[(3) - (4)].t_exp), false, &vars); }
    break;

  case 15:

/* Line 1806 of yacc.c  */
#line 76 "MiniMatlab/parser.ypp"
    { (yyval.t_stmt) = new VarSliceStmt((yyvsp[(1) - (4)].t_varSl), (yyvsp[(3) - (4)].t_exp), false, &vars); }
    break;

  case 16:

/* Line 1806 of yacc.c  */
#line 77 "MiniMatlab/parser.ypp"
    { (yyval.t_stmt) = (yyvsp[(1) - (1)].t_stmt); }
    break;

  case 17:

/* Line 1806 of yacc.c  */
#line 78 "MiniMatlab/parser.ypp"
    { (yyval.t_stmt) = NULL; }
    break;

  case 18:

/* Line 1806 of yacc.c  */
#line 82 "MiniMatlab/parser.ypp"
    { (yyval.t_array) = new NArray(); (yyval.t_array)->add((yyvsp[(1) - (1)].t_exp)); }
    break;

  case 19:

/* Line 1806 of yacc.c  */
#line 83 "MiniMatlab/parser.ypp"
    { (yyvsp[(1) - (3)].t_array)->add((yyvsp[(3) - (3)].t_exp)); }
    break;

  case 20:

/* Line 1806 of yacc.c  */
#line 87 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new PlusExp((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 21:

/* Line 1806 of yacc.c  */
#line 88 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new MinusExp((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 22:

/* Line 1806 of yacc.c  */
#line 89 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(1) - (1)].t_exp); }
    break;

  case 23:

/* Line 1806 of yacc.c  */
#line 92 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new MultExp((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 24:

/* Line 1806 of yacc.c  */
#line 93 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new DivExp((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 25:

/* Line 1806 of yacc.c  */
#line 94 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(1) - (1)].t_exp); }
    break;

  case 26:

/* Line 1806 of yacc.c  */
#line 97 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(2) - (2)].t_exp); }
    break;

  case 27:

/* Line 1806 of yacc.c  */
#line 98 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new MinusNumber((yyvsp[(2) - (2)].t_exp)); }
    break;

  case 28:

/* Line 1806 of yacc.c  */
#line 99 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(1) - (1)].t_exp); }
    break;

  case 29:

/* Line 1806 of yacc.c  */
#line 102 "MiniMatlab/parser.ypp"
    { (yyval.t_slice) = new Slice(new Number(1), new Number(1), new Number(0)); }
    break;

  case 30:

/* Line 1806 of yacc.c  */
#line 103 "MiniMatlab/parser.ypp"
    { (yyval.t_slice) = new Slice((yyvsp[(1) - (1)].t_exp), new Number(1), (yyvsp[(1) - (1)].t_exp)); }
    break;

  case 31:

/* Line 1806 of yacc.c  */
#line 104 "MiniMatlab/parser.ypp"
    { (yyval.t_slice) = new Slice((yyvsp[(1) - (3)].t_exp), new Number(1), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 32:

/* Line 1806 of yacc.c  */
#line 105 "MiniMatlab/parser.ypp"
    { (yyval.t_slice) = new Slice((yyvsp[(1) - (5)].t_exp), (yyvsp[(3) - (5)].t_exp), (yyvsp[(5) - (5)].t_exp)); }
    break;

  case 33:

/* Line 1806 of yacc.c  */
#line 108 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new PlusExp((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 34:

/* Line 1806 of yacc.c  */
#line 109 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new MinusExp((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 35:

/* Line 1806 of yacc.c  */
#line 110 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(1) - (1)].t_exp); }
    break;

  case 36:

/* Line 1806 of yacc.c  */
#line 113 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new MultExp((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 37:

/* Line 1806 of yacc.c  */
#line 114 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new DivExp((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 38:

/* Line 1806 of yacc.c  */
#line 115 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(1) - (1)].t_exp); }
    break;

  case 39:

/* Line 1806 of yacc.c  */
#line 118 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(2) - (2)].t_exp); }
    break;

  case 40:

/* Line 1806 of yacc.c  */
#line 119 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new MinusNumber((yyvsp[(2) - (2)].t_exp)); }
    break;

  case 41:

/* Line 1806 of yacc.c  */
#line 120 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(1) - (1)].t_exp); }
    break;

  case 42:

/* Line 1806 of yacc.c  */
#line 123 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new Number((yyvsp[(1) - (1)].num)); }
    break;

  case 43:

/* Line 1806 of yacc.c  */
#line 124 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new Variable((yyvsp[(1) - (1)].name), &vars); }
    break;

  case 44:

/* Line 1806 of yacc.c  */
#line 125 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(2) - (3)].t_exp); }
    break;

  case 45:

/* Line 1806 of yacc.c  */
#line 126 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new Number(0); }
    break;

  case 46:

/* Line 1806 of yacc.c  */
#line 129 "MiniMatlab/parser.ypp"
    { (yyval.t_varSl) = new VarSl((yyvsp[(1) - (4)].name), (yyvsp[(3) - (4)].t_slice), &vars); }
    break;

  case 47:

/* Line 1806 of yacc.c  */
#line 130 "MiniMatlab/parser.ypp"
    { (yyval.t_varSl) = new VarSl((yyvsp[(1) - (6)].name), (yyvsp[(3) - (6)].t_slice), (yyvsp[(5) - (6)].t_slice), &vars); }
    break;

  case 48:

/* Line 1806 of yacc.c  */
#line 133 "MiniMatlab/parser.ypp"
    { (yyval.t_index) = new Index((yyvsp[(1) - (1)].t_exp), (yyvsp[(1) - (1)].t_exp)); }
    break;

  case 49:

/* Line 1806 of yacc.c  */
#line 134 "MiniMatlab/parser.ypp"
    { (yyval.t_index) = new Index((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 50:

/* Line 1806 of yacc.c  */
#line 137 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new PlusExp((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 51:

/* Line 1806 of yacc.c  */
#line 138 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new MinusExp((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 52:

/* Line 1806 of yacc.c  */
#line 139 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(1) - (1)].t_exp); }
    break;

  case 53:

/* Line 1806 of yacc.c  */
#line 142 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new MultExp((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 54:

/* Line 1806 of yacc.c  */
#line 143 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new DivExp((yyvsp[(1) - (3)].t_exp), (yyvsp[(3) - (3)].t_exp)); }
    break;

  case 55:

/* Line 1806 of yacc.c  */
#line 144 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(1) - (1)].t_exp); }
    break;

  case 56:

/* Line 1806 of yacc.c  */
#line 147 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(2) - (2)].t_exp); }
    break;

  case 57:

/* Line 1806 of yacc.c  */
#line 148 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new MinusNumber((yyvsp[(2) - (2)].t_exp)); }
    break;

  case 58:

/* Line 1806 of yacc.c  */
#line 149 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(1) - (1)].t_exp); }
    break;

  case 59:

/* Line 1806 of yacc.c  */
#line 152 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new Number((yyvsp[(1) - (1)].num)); }
    break;

  case 60:

/* Line 1806 of yacc.c  */
#line 153 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new Variable((yyvsp[(1) - (1)].name), &vars); }
    break;

  case 61:

/* Line 1806 of yacc.c  */
#line 154 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(2) - (3)].t_exp); }
    break;

  case 62:

/* Line 1806 of yacc.c  */
#line 157 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new Number((yyvsp[(1) - (1)].num)); }
    break;

  case 63:

/* Line 1806 of yacc.c  */
#line 158 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new Variable((yyvsp[(1) - (1)].name), &vars); }
    break;

  case 64:

/* Line 1806 of yacc.c  */
#line 159 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(1) - (1)].t_varSl); }
    break;

  case 65:

/* Line 1806 of yacc.c  */
#line 160 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(2) - (3)].t_array); }
    break;

  case 66:

/* Line 1806 of yacc.c  */
#line 161 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = (yyvsp[(2) - (3)].t_exp); }
    break;

  case 67:

/* Line 1806 of yacc.c  */
#line 162 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new Random((yyvsp[(3) - (4)].t_index)); }
    break;

  case 68:

/* Line 1806 of yacc.c  */
#line 164 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new Zeros((yyvsp[(3) - (4)].t_index)); }
    break;

  case 69:

/* Line 1806 of yacc.c  */
#line 166 "MiniMatlab/parser.ypp"
    { (yyval.t_exp) = new Ones((yyvsp[(3) - (4)].t_index)); }
    break;



/* Line 1806 of yacc.c  */
#line 1995 "MiniMatlab/.tab.cpp"
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 2067 of yacc.c  */
#line 170 "MiniMatlab/parser.ypp"

//-- FUNCTION DEFINITIONS ---------------------------------
void clean_up() {
	cout << "Clean Up!\n";
	map<string, multi_array<double>* >::iterator it;
	for (it = vars.begin(); it != vars.end(); ++it) {
		cout << it->first << " ";
  		delete it->second;
	}
	cout << "\n";
}

int main(int argc, char *argv[]) {
  	if (argc == 2) {
		FILE *myfile = fopen(argv[1], "r");
		if (!(yyin = myfile)) {
			cout << "Error occured when attempting to open: " 
				<< argv[0] << "!\n";
			return -1;
		}

		yyparse();
		fclose(yyin);
		clean_up();

	} else if (argc == 1) {
		cout << "Welcome to MicroMatlab 1.0, exit by pressing <ctrl-d>.\n\n"; 
		yyparse();
		clean_up();
	} else {
		cout << "Usage: microMatlab filename\n";
	}

	//ast->evaluate();

	return 0;
}

