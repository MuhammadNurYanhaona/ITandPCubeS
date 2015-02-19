#ifndef _H_sync
#define _H_sync

/*
==================================================
October 13, 2014
Author: Andrew Grimshaw
Purpose: Implement Barriers an other synchronization mechanisms required for IT.
Copyright 2015 by Andrew Grimshaw

==================================================
*/

#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>

class Barrier {
	// How many threads need call wait before releasing all threads
	int _size, _count;
	sem_t mutex;  	// the mutex
	sem_t throttle;	// Waiters signal the releaser so that there is no over-pumping
	sem_t waitq;	// The semaphore on which the waiters wait
	
public:
	Barrier(int size);
	void wait();
};

class RS {
	// How many threads need call wait before releasing all threads
	int _size, _count,_iteration;
// _iteration is the last iteration for which a signal has been received.
	int _gappers;
	sem_t mutex;  	// the mutex
	sem_t throttle;	// Waiters signal the releaser so that there is no over-pumping
	sem_t throttle2;
	// May be able to modify the implementation into a single counting semaphore
	sem_t waitq;	// The semaphore on which the waiters wait
	Barrier b;
	
public:
	RS(int size);
	void wait(int iteration);
	void signal(int iteration);
};

#endif
