
// This is a guard condition so that contents of this file are not included
// more than once.

#ifndef HEADER_H
#define	HEADER_H

#include <xc.h> // include processor files - each processor file is guarded.
#include "mcc_generated_files/pin_manager.h"
#include <stdint.h>
#include <stdbool.h>

#define _XTAL_FREQ 250000
#define DEBOUNCEMS 20

#define N0  0b0000
#define N1  0b0100                     //bin to dice
#define N2  0b0010
#define N3  0b0110
#define N4  0b0001
#define N5  0b0101
#define N6  0b0011
#define N7  0b0111

#define driveLED(numberIndex)    (LATA = validNumbers[numberIndex])
#define clearLED()               (LATA = N0)
#define buttonPressed            (PORTAbits.RA3 == 1)

unsigned int counter;

struct DICE {
   unsigned int next : 4; //4 bit for each number
   unsigned int current : 4;
};

struct DICE dice;

char validNumbers[6] = {N1, N2, N3, N4, N5, N6};

//extern void driveLed(char);       //Routine for Simulation of Throwing a Dice
extern char throwDice(void);  //Routine for Animation while throwing process

#endif
