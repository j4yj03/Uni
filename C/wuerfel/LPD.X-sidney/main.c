/**
  Generated Main Source File

  Authors:
 * Daniel 'Teps' Haubenreißer, Sidney Göhler, Nawar Maafa

  University:
    HTW Berlin

  File Name:
    main.c

  Summary:
    This is the main file for the electrical dice

  Description:
    This header file provides implementations for driver APIs for all modules selected in the GUI.
    Generation Information :
        Product Revision  :  PIC10 / PIC12 / PIC16 / PIC18 MCUs - 1.81.6
        Device            :  PIC10LF320
        Driver Version    :  2.00
*/

#include "mcc_generated_files/mcc.h"
#include "header.h"

// CONFIG
#pragma config FOSC = INTOSC    // Oscillator Selection bits->INTOSC oscillator: CLKIN function disabled
#pragma config BOREN = OFF    // Brown-out Reset Enable->Brown-out Reset enabled
#pragma config WDTE = OFF    // Watchdog Timer Enable->WDT disabled
#pragma config PWRTE = OFF    // Power-up Timer Enable bit->PWRT disabled
#pragma config MCLRE = OFF   // MCLR Pin Function Select bit->MCLR pin function is MCLR
#pragma config CP = OFF    // Code Protection bit->Program memory code protection is disabled
#pragma config LVP = OFF    // Low-Voltage Programming Enable->Low-voltage programming enabled
#pragma config LPBOR = OFF    // Brown-out Reset Selection bits->BOR enabled
#pragma config BORV = LO    // Brown-out Reset Voltage Selection->Brown-out Reset Voltage (Vbor), low trip point selected.
#pragma config WRT = OFF    // Flash Memory Self-Write Protection->Write protection off

/*
                         Main application
 */
void main(void)
{
    // initialize the device
    SYSTEM_Initialize();
    
    

    // Enable the Global Interrupts
    INTERRUPT_GlobalInterruptEnable();

    // Enable the Peripheral Interrupts
    //INTERRUPT_PeripheralInterruptEnable();

    // Disable the Global Interrupts
    //INTERRUPT_GlobalInterruptDisable();

    // Disable the Peripheral Interrupts
    //INTERRUPT_PeripheralInterruptDisable();

    while (1)
    {
        counter=0;
        SLEEP();
        
    }
}
/**
 End of File
*/