#include "extern_config.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/string_utils.h"

#include <cstdlib>
#include <iostream>

//---------------------------------------------------------- Language Includes and Links -------------------------------------------------------/

LanguageIncludesAndLinks::LanguageIncludesAndLinks(const char *language) {
	this->language = language;
	this->headerIncludes = new List<const char*>;
	this->libraryLinks = new List<const char*>;
}

void LanguageIncludesAndLinks::addHeaders(List<const char*> *newHeaders) {
	string_utils::combineLists(headerIncludes, newHeaders);
}

void LanguageIncludesAndLinks::addLibraryLinks(List<const char*> *newLinks) {
	string_utils::combineLists(libraryLinks, newLinks);
}

//------------------------------------------------------------ Includes and Links Map ----------------------------------------------------------/

IncludesAndLinksMap::IncludesAndLinksMap() {
	map = new Hashtable<LanguageIncludesAndLinks*>;
	languagesUsed = new List<const char*>;
}

bool IncludesAndLinksMap::hasExternBlocksForLanguage(const char *language) {
	return (map->Lookup(language) != NULL);
}

void IncludesAndLinksMap::addIncludesAndLinksForLanguage(const char *language, 
		List<const char*> *includes, List<const char*> *links) {
	
	LanguageIncludesAndLinks *current = map->Lookup(language);
	if (current == NULL) {
		languagesUsed->Append(language);
		current = new LanguageIncludesAndLinks(language);
	}

	if (includes != NULL) {
		current->addHeaders(includes);
	}
	if (links != NULL) {
		current->addLibraryLinks(links);
	}
	map->Enter(language, current);
}
