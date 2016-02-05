scanner: lex.yy.o symtab.c
	gcc lex.yy.o symtab.c -lfl -o scanner 

lex.yy.o:  lex.yy.c token.h symtab.h
	gcc -c lex.yy.c

lex.yy.c: IT.l
	flex -d IT.l
clean:
	rm -f scanner lex.* 
