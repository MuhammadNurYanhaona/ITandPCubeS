/*
 * structures.h
 *
 *  Created on: Dec 14, 2014
 *      Author: yan
 */

#ifndef STRUCTURES_H_
#define STRUCTURES_H_

typedef struct {
        int min;
        int max;
} Range;

typedef struct {
        int length;
        Range range;
} Dimension;

typedef struct {
        int id;
        const char *name;
        int units;
} PPS_Definition;

typedef struct {
        char LPS;
        PPS_Definition *PPS;
} MapEntry;


#endif /* STRUCTURES_H_ */
