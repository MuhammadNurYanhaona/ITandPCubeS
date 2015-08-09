#ifndef _H_list
#define _H_list

#include <deque>

template<class Element> class List {

private:
	std::deque<Element> elems;

public:
	// Create a new empty list
	List() {}

	// Create a list with an initial capacity
	List(int initialCapacity) {}

	// Returns count of elements currently in list
	int NumElements() const
	{ return elems.size(); }

	// Returns element at index in list. Indexing is 0-based.
	// Raises an assert if index is out of range.
	Element Nth(int index) const
	{
		return elems[index]; }

	// Inserts element at index, shuffling over others
	// Raises assert if index out of range
	void InsertAt(const Element &elem, int index)
	{ elems.insert(elems.begin() + index, elem); }

	// Adds element to list end
	void Append(const Element &elem)
	{ elems.push_back(elem); }

	void AppendAll(List<Element> *elements) {
		for (int i = 0; i < elements->NumElements(); i++)
			Append(elements->Nth(i));
	}

	// Removes element at index, shuffling down others
	// Raises assert if index out of range
	void RemoveAt(int index)
	{ elems.erase(elems.begin() + index); }

};

#endif
