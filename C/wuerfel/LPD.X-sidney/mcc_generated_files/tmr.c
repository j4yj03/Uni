/**
  TMR2 Generated Driver File

  @Company
    Microchip Technology Inc.

  @File Name
    tmr2.c

  @Summary
    This is the generated driver implementation file for the TMR2 driver using PIC10 / PIC12 / PIC16 / PIC18 MCUs

  @Description
    This source file provides APIs for TMR2.
    Generation Information :
        Product Revision  :  PIC10 / PIC12 / PIC16 / PIC18 MCUs - 1.81.6
        Device            :  PIC10LF320
        Driver Version    :  2.01
    The generated drivers are tested against the following:
        Compiler          :  XC8 2.30 and above
        MPLAB 	          :  MPLAB X 5.40
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

/**
  Section: Included Files
*/

#include <xc.h>
#include "tmr.h"
#include "../header.h"
/**
  Section: Global Variables Definitions
*/


/**
  Section: TMR2 APIs
*/

void TIMER_Initialize(void)
{
    // Set TMR2
    // PR2 1;
    // PR2 = 0x00;
    PR2 = 0x01;
    // PR2 244;
    //PR2 = 0xF4;

    // TMR2 0;
    TMR2 = 0x00;

    // Clearing IF flag before enabling the interrupt.
    PIR1bits.TMR2IF = 0;

    // Enabling TMR2 interrupt.
    PIE1bits.TMR2IE = 1;

    // Interrupt Handler in interrupt_manager.c

    // T2CKPS 1:1; TOUTPS 1:1; TMR2ON on;
    //T2CON = 0x04;
    // T2CKPS 1:1; TOUTPS 1:1; TMR2ON off;
    //T2CON = 0x00;
     // T2CKPS 1:64; TOUTPS 1:8; TMR2ON on;
    //T2CON = 0x3F;
    // T2CKPS 1:64; TOUTPS 1:8; TMR2ON 0ff;
    T2CON = 0x3B


    ///////////////////////////////////////////////////////////////////////////
    // Set TMR0

    PR0 = 0x01

    TMR0 = 0x00

    // Clearing IF flag before enabling the interrupt.
    PIR1bits.TMR0IF = 0;

    // Enabling TMR0 interrupt.
    PIE1bits.TMR0IE = 0;

    T0CON = 0x00

}

void TMR2_StartTimer(void)
{

    TMR2 = 0x00; //reset timer

    // Start the Timer by writing to TMRxON bit
    T2CONbits.TMR2ON = 1;

}

void TMR2_StopTimer(void)
{
    // Stop the Timer by writing to TMRxON bit
    T2CONbits.TMR2ON = 0;
}

unsigned char TMR0_ReadTimer(void)
{
    unsigned char readVal;

    readVal = TMR0;

    return readVal;
}



/**
  End of File
*/
