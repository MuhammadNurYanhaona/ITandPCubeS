#ifndef _H_list
#define _H_list

#include <iostream>
#include <deque>
  
class Node;

template<class Element> class List {

 private:
    std::deque<Element> elems;

 public:
    	// Create a new empty list
    	List() {}
    	~List() { elems.clear(); }
	// create a list with an initial capacity (need to be properly implemented later during list refactoring)	
	List(int capacity) {}

        // Returns count of elements currently in list
    	int NumElements() const { 
		return elems.size(); 
	}

        // Returns element at index in list. Indexing is 0-based.
        // Raises an assert if index is out of range.
    	Element Nth(int index) const { 
	  	return elems[index]; 
	}

        // Inserts element at index, shuffling over others
   	// Raises assert if index out of range
    	void InsertAt(const Element &elem, int index) { 
	  	elems.insert(elems.begin() + index, elem); 
	}

        // Adds element to list end
    	void Append(const Element &elem) { elems.push_back(elem); }

    	void AppendAll(List<Element> *elements) {
		for (int i = 0; i < elements->NumElements(); i++) Append(elements->Nth(i));
    	}	

        // Removes element at index, shuffling down others
        // Raises assert if index out of range
    	void RemoveAt(int index) { 
	  	elems.erase(elems.begin() + index); 
	}
          
       	// These are some specific methods useful for lists of ast nodes
       	// They will only work on lists of elements that respond to the
       	// messages, but since C++ only instantiates the template if you use
       	// you can still have Lists of ints, chars*, as long as you 
       	// don't try to SetParentAll on that list.
    	void SetParentAll(Node *p) { 
		for (int i = 0; i < NumElements(); i++) Nth(i)->SetParent(p); 
	}
    	
	void PrintAll(int indentLevel, const char *label = NULL) { 
		for (int i = 0; i < NumElements(); i++) Nth(i)->Print(indentLevel, label); 
	}
             
	// Removes all elements from the list
	void clear() {
		while (elems.size() > 0) elems.erase(elems.begin());
	}
};

#endif

