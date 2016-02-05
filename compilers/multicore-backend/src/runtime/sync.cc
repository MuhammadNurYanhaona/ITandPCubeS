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
#include "sync.h"


Barrier::Barrier(int size) {
        _size=size;
        _count=size;
        sem_init(&mutex,0,1);   // init to 1
        sem_init(&throttle,0,0);        // init to 0
        sem_init(&waitq,0,0);   // init to 0
}

void Barrier::wait(){
	sem_wait(&mutex);	// Make sure only one in at a  time
	_count--;
	if (_count==0 ) {
		// Time to wake everyone up
		for (int i=1;i<_size;i++) {
			sem_post(&waitq);	// Wake up another waiter
			sem_wait(&throttle);	// Wait for the waiter to awaken and signal me.
		}
		_count=_size;	// Reset the counter
		sem_post(&mutex);	// Release the mutex
	}
	else {
		// Block myself, but first release mutex
		sem_post(&mutex);	
		sem_wait(&waitq);	// Sleep
		sem_post(&throttle);	// wake up the releaser
	}
}


RS::RS(int size): b(size) {
	_size=size;
	_count=0;
	_gappers=0;
	_iteration=-1;
	sem_init(&mutex,0,1); 	// init to 1
	sem_init(&throttle,0,0);	// init to 0
	sem_init(&throttle2,0,0);	// init to 0
	sem_init(&waitq,0,0);	// init to 0
}


void RS::signal(int iteration){
	b.wait();return;
}

void RS::wait(int iteration){
	b.wait();return;
}

