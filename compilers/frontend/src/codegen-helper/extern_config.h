#ifndef _H_extern_config
#define _H_extern_config

/* This header file contains classes that hold information about different extern code blocks present in an IT program. 
 */

#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

// This class provides a summary of the header files to be included and libraries to be linked for all extern code
// blocks of a task that are written using a particular language.
class LanguageIncludesAndLinks {
  protected:
	const char *language;
	List<const char*> *headerIncludes;
	List<const char*> *libraryLinks;
  public:
	LanguageIncludesAndLinks(const char *language);
	void addHeaders(List<const char*> *newHeaders);
	void addLibraryLinks(List<const char*> *newLinks);
	List<const char*> *getHeaderIncludes() { return headerIncludes; }
	List<const char*> *getLibraryLinks() { return libraryLinks; }
};

// This class holds information about all header includes and library links of all languages found within a IT task.
class IncludesAndLinksMap {
  protected:
	Hashtable<LanguageIncludesAndLinks*> *map;
	List<const char*> *languagesUsed;
  public:
	IncludesAndLinksMap();
	bool hasExternBlocksForLanguage(const char *language);
	LanguageIncludesAndLinks *getIncludesAndLinksForLanguage(const char *language) {
		return map->Lookup(language);
	}
	void addIncludesAndLinksForLanguage(const char *language, 
			List<const char*> *includes, List<const char*> *links);
	List<const char*> *getLanguagesUsed() { return languagesUsed; }
};

#endif
