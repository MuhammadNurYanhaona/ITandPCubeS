/* File: errors.h
 * --------------
 * This file defines an error-reporting class with a set of already
 * implemented static methods for reporting the standard IT errors.
 */

#ifndef _H_errors
#define _H_errors

#include <string>
using std::string;
#include "location.h"

class ReportError
{
 public:
  // Generic method to report a printf-style error message
  static void Formatted(yyltype *loc, const char *format, ...);
  // Returns number of error messages printed
  static int NumErrors() { return numErrors; }
  
 private:

  static void UnderlineErrorInLine(const char *line, yyltype *pos);
  static void OutputError(yyltype *loc, string msg);
  static int numErrors;
  
};

#endif
