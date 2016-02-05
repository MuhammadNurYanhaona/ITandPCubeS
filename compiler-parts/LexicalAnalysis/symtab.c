
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "symtab.h"


#define HSIZE 211

static int level=0;
static node *table[HSIZE] = {0};

static unsigned int hvalue(char *s)
{ unsigned int i=0;
  while (*s) i= (i<<2)+*s++;
  return i % HSIZE;
}

static node *newnode(char *s)
{ node *n;
  n = (node *)malloc(sizeof(*n));
  assert(n!=NULL);
  n->next=NULL;
  n->symbol = (char *)malloc(strlen(s)+1);
  assert(n->symbol!=NULL);
  strcpy(n->symbol,s);
  n->level=level;
  return n;
}

node *insert(char *s)
{ node *n;
  unsigned int i;
  n = newnode(s);
  i = hvalue(s);
  n->next=table[i];
  table[i] = n;
  return n;
}



node *lookup(char *s)
{ node *n;
  n = table[hvalue(s)];
  while (n!=NULL)
    if (strcmp(s,n->symbol)==0)
      return n;
    else
      n=n->next;   
  return n;
}


node *delete(char *s)
{ node **p;
  p = table+ hvalue(s);
  while (*p!=NULL)
    if (strcmp(s,(*p)->symbol)==0)
    { node *n;
      n=*p;
      *p=(*p)->next;
      return n;
    }
    else
      p = &((*p)->next);
   return NULL;
}   

void scope_open(void)
{ level++;
}

void scope_close(void)
{ int i;
  level--;
  for (i=0; i< HSIZE; i++)
    while (table[i]!=NULL && table[i]->level>level)
    { node *n;
      n=table[i];
      table[i]=n->next;
      free(n);
    }
}
