/**
  @Generated PIC10 / PIC12 / PIC16 / PIC18 MCUs Source File

  @Company:
    Microchip Technology Inc.

  @File Name:
    mcc.c

  @Summary:
    This is the mcc.c file generated using PIC10 / PIC12 / PIC16 / PIC18 MCUs

  @Description:
    This header file provides implementations for driver APIs for all modules selected in the GUI.
    Generation Information :
        Product Revision  :  PIC10 / PIC12 / PIC16 / PIC18 MCUs - 1.81.6
        Device            :  PIC10LF320
        Driver Version    :  2.00
    The generated drivers are tested against the following:
        Compiler          :  XC8 2.30 and above or later
        MPLAB             :  MPLAB X 5.40
*/

/*
    (c) 2018 Microchip Technology Inc. and its subsidiaries.

    Subject to your compliance with these terms, you may use Microchip software and any
    derivatives exclusively with Microchip products. It is your responsibility to comply with third party
    license terms applicable to your use of third party software (including open source software) that
    may accompany Microchip software.

    THIS SOFTWARE IS SUPPLIED BY MICROCHIP "AS IS". NO WARRANTIES, WHETHER
    EXPRESS, IMPLIED OR STATUTORY, APPLY TO THIS SOFTWARE, INCLUDING ANY
    IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, AND FITNESS
    FOR A PARTICULAR PURPOSE.

    IN NO EVENT WILL MICROCHIP BE LIABLE FOR ANY INDIRECT, SPECIAL, PUNITIVE,
    INCIDENTAL OR CONSEQUENTIAL LOSS, DAMAGE, COST OR EXPENSE OF ANY KIND
    WHATSOEVER RELATED TO THE SOFTWARE, HOWEVER CAUSED, EVEN IF MICROCHIP
    HAS BEEN ADVISED OF THE POSSIBILITY OR THE DAMAGES ARE FORESEEABLE. TO
    THE FULLEST EXTENT ALLOWED BY LAW, MICROCHIP'S TOTAL LIABILITY ON ALL
    CLAIMS IN ANY WAY RELATED TO THIS SOFTWARE WILL NOT EXCEED THE AMOUNT
    OF FEES, IF ANY, THAT YOU HAVE PAID DIRECTLY TO MICROCHIP FOR THIS
    SOFTWARE.
*/

#include "mcc.h"
#include "../header.h"

#if (__XC8_VERSION < 1430)
asm("psect functab,global,class=CODE,reloc=0x1,delta=2");
#endif // __XC8_VERSION

void SYSTEM_Initialize(void)
{
    //initialize
    //counter = 0; //Counter Variable for Interrupt Handling
    //throw_check = 0; //Check whether Dice is thrown or old value should be displayed
    //randn = 0; //Randomly generated number which will be converted to a binary value for driving the LEDs
    //new_val = 0;  //If Dice is thrown create a New_Value
    dice.current = 0;   //current number of the dicerole
    dice.next = 0;    //next roled number for the dice

    PIN_MANAGER_Initialize();
    OSCILLATOR_Initialize();
    WDT_Initialize();     //do we need the watchdooogee?
    TIMER_Initialize();
}

void OSCILLATOR_Initialize(void)
{
    // LFIOFR 31.25KHz_osc_not_ready; HFIOFS unstable; HFIOFR 16MHz_osc_not_ready; IRCF 250KHz;
    OSCCON = 0x10;
    // CLKROE disabled;
    CLKRCON = 0x00;
    // SBOREN disabled; BORFS disabled; BORRDY BOR Circuit is inactive;
    BORCON = 0x00;
}

void WDT_Initialize(void)
{
    // WDTPS 1:65536; SWDTEN OFF;
    WDTCON = 0x16;
}

/**
 End of File
*/
