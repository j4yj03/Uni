# 1 "mcc_generated_files/pin_manager.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 288 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\language_support.h" 1 3
# 2 "<built-in>" 2
# 1 "mcc_generated_files/pin_manager.c" 2
# 49 "mcc_generated_files/pin_manager.c"
# 1 "mcc_generated_files/pin_manager.h" 1
# 54 "mcc_generated_files/pin_manager.h"
# 1 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\xc.h" 1 3
# 18 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\xc.h" 3
extern const char __xc8_OPTIM_SPEED;

extern double __fpnormalize(double);



# 1 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\xc8debug.h" 1 3
# 13 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\xc8debug.h" 3
#pragma intrinsic(__builtin_software_breakpoint)
extern void __builtin_software_breakpoint(void);
# 24 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\xc.h" 2 3



# 1 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\pic.h" 1 3




# 1 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\htc.h" 1 3



# 1 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\xc.h" 1 3
# 5 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\htc.h" 2 3
# 6 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\pic.h" 2 3







# 1 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\pic_chip_select.h" 1 3
# 114 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\pic_chip_select.h" 3
# 1 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 1 3
# 44 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
# 1 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\__at.h" 1 3
# 45 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 2 3







extern volatile unsigned char INDF __attribute__((address(0x000)));

__asm("INDF equ 00h");




extern volatile unsigned char TMR0 __attribute__((address(0x001)));

__asm("TMR0 equ 01h");




extern volatile unsigned char PCL __attribute__((address(0x002)));

__asm("PCL equ 02h");




extern volatile unsigned char STATUS __attribute__((address(0x003)));

__asm("STATUS equ 03h");


typedef union {
    struct {
        unsigned C :1;
        unsigned DC :1;
        unsigned Z :1;
        unsigned nPD :1;
        unsigned nTO :1;
        unsigned RP0 :1;
        unsigned RP1 :1;
        unsigned IRP :1;
    };
    struct {
        unsigned CARRY :1;
        unsigned :1;
        unsigned ZERO :1;
    };
} STATUSbits_t;
extern volatile STATUSbits_t STATUSbits __attribute__((address(0x003)));
# 150 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char FSR __attribute__((address(0x004)));

__asm("FSR equ 04h");




extern volatile unsigned char PORTA __attribute__((address(0x005)));

__asm("PORTA equ 05h");


typedef union {
    struct {
        unsigned RA0 :1;
        unsigned RA1 :1;
        unsigned RA2 :1;
        unsigned RA3 :1;
    };
} PORTAbits_t;
extern volatile PORTAbits_t PORTAbits __attribute__((address(0x005)));
# 195 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char TRISA __attribute__((address(0x006)));

__asm("TRISA equ 06h");


typedef union {
    struct {
        unsigned TRISA0 :1;
        unsigned TRISA1 :1;
        unsigned TRISA2 :1;
    };
} TRISAbits_t;
extern volatile TRISAbits_t TRISAbits __attribute__((address(0x006)));
# 227 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char LATA __attribute__((address(0x007)));

__asm("LATA equ 07h");


typedef union {
    struct {
        unsigned LATA0 :1;
        unsigned LATA1 :1;
        unsigned LATA2 :1;
    };
} LATAbits_t;
extern volatile LATAbits_t LATAbits __attribute__((address(0x007)));
# 259 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char ANSELA __attribute__((address(0x008)));

__asm("ANSELA equ 08h");


typedef union {
    struct {
        unsigned ANSA0 :1;
        unsigned ANSA1 :1;
        unsigned ANSA2 :1;
    };
} ANSELAbits_t;
extern volatile ANSELAbits_t ANSELAbits __attribute__((address(0x008)));
# 291 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char WPUA __attribute__((address(0x009)));

__asm("WPUA equ 09h");


typedef union {
    struct {
        unsigned WPUA0 :1;
        unsigned WPUA1 :1;
        unsigned WPUA2 :1;
        unsigned WPUA3 :1;
    };
} WPUAbits_t;
extern volatile WPUAbits_t WPUAbits __attribute__((address(0x009)));
# 329 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PCLATH __attribute__((address(0x00A)));

__asm("PCLATH equ 0Ah");


typedef union {
    struct {
        unsigned PCLH0 :1;
    };
} PCLATHbits_t;
extern volatile PCLATHbits_t PCLATHbits __attribute__((address(0x00A)));
# 349 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char INTCON __attribute__((address(0x00B)));

__asm("INTCON equ 0Bh");


typedef union {
    struct {
        unsigned IOCIF :1;
        unsigned INTF :1;
        unsigned TMR0IF :1;
        unsigned IOCIE :1;
        unsigned INTE :1;
        unsigned TMR0IE :1;
        unsigned PEIE :1;
        unsigned GIE :1;
    };
} INTCONbits_t;
extern volatile INTCONbits_t INTCONbits __attribute__((address(0x00B)));
# 411 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PIR1 __attribute__((address(0x00C)));

__asm("PIR1 equ 0Ch");


typedef union {
    struct {
        unsigned :1;
        unsigned TMR2IF :1;
        unsigned :1;
        unsigned CLC1IF :1;
        unsigned NCO1IF :1;
        unsigned :1;
        unsigned ADIF :1;
    };
} PIR1bits_t;
extern volatile PIR1bits_t PIR1bits __attribute__((address(0x00C)));
# 452 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PIE1 __attribute__((address(0x00D)));

__asm("PIE1 equ 0Dh");


typedef union {
    struct {
        unsigned :1;
        unsigned TMR2IE :1;
        unsigned :1;
        unsigned CLC1IE :1;
        unsigned NCO1IE :1;
        unsigned :1;
        unsigned ADIE :1;
    };
} PIE1bits_t;
extern volatile PIE1bits_t PIE1bits __attribute__((address(0x00D)));
# 493 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char OPTION_REG __attribute__((address(0x00E)));

__asm("OPTION_REG equ 0Eh");


typedef union {
    struct {
        unsigned PS :3;
        unsigned PSA :1;
        unsigned T0SE :1;
        unsigned T0CS :1;
        unsigned INTEDG :1;
        unsigned nWPUEN :1;
    };
    struct {
        unsigned PS0 :1;
        unsigned PS1 :1;
        unsigned PS2 :1;
    };
} OPTION_REGbits_t;
extern volatile OPTION_REGbits_t OPTION_REGbits __attribute__((address(0x00E)));
# 563 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PCON __attribute__((address(0x00F)));

__asm("PCON equ 0Fh");


typedef union {
    struct {
        unsigned nBOR :1;
        unsigned nPOR :1;
    };
} PCONbits_t;
extern volatile PCONbits_t PCONbits __attribute__((address(0x00F)));
# 589 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char OSCCON __attribute__((address(0x010)));

__asm("OSCCON equ 010h");


typedef union {
    struct {
        unsigned HFIOFS :1;
        unsigned LFIOFR :1;
        unsigned :1;
        unsigned HFIOFR :1;
        unsigned IRCF :3;
    };
    struct {
        unsigned :4;
        unsigned IRCF0 :1;
        unsigned IRCF1 :1;
        unsigned IRCF2 :1;
    };
} OSCCONbits_t;
extern volatile OSCCONbits_t OSCCONbits __attribute__((address(0x010)));
# 649 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char TMR2 __attribute__((address(0x011)));

__asm("TMR2 equ 011h");




extern volatile unsigned char PR2 __attribute__((address(0x012)));

__asm("PR2 equ 012h");




extern volatile unsigned char T2CON __attribute__((address(0x013)));

__asm("T2CON equ 013h");


typedef union {
    struct {
        unsigned T2CKPS :2;
        unsigned TMR2ON :1;
        unsigned TOUTPS :4;
    };
    struct {
        unsigned T2CKPS0 :1;
        unsigned T2CKPS1 :1;
        unsigned :1;
        unsigned TOUTPS0 :1;
        unsigned TOUTPS1 :1;
        unsigned TOUTPS2 :1;
        unsigned TOUTPS3 :1;
    };
} T2CONbits_t;
extern volatile T2CONbits_t T2CONbits __attribute__((address(0x013)));
# 734 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PWM1DCL __attribute__((address(0x014)));

__asm("PWM1DCL equ 014h");


typedef union {
    struct {
        unsigned :6;
        unsigned PWM1DCL :2;
    };
    struct {
        unsigned :6;
        unsigned PWM1DCL0 :1;
        unsigned PWM1DCL1 :1;
    };
} PWM1DCLbits_t;
extern volatile PWM1DCLbits_t PWM1DCLbits __attribute__((address(0x014)));
# 770 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PWM1DCH __attribute__((address(0x015)));

__asm("PWM1DCH equ 015h");


typedef union {
    struct {
        unsigned PWM1DCH :8;
    };
    struct {
        unsigned PWM1DCH0 :1;
        unsigned PWM1DCH1 :1;
        unsigned PWM1DCH2 :1;
        unsigned PWM1DCH3 :1;
        unsigned PWM1DCH4 :1;
        unsigned PWM1DCH5 :1;
        unsigned PWM1DCH6 :1;
        unsigned PWM1DCH7 :1;
    };
} PWM1DCHbits_t;
extern volatile PWM1DCHbits_t PWM1DCHbits __attribute__((address(0x015)));
# 840 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PWM1CON __attribute__((address(0x016)));

__asm("PWM1CON equ 016h");


extern volatile unsigned char PWM1CON0 __attribute__((address(0x016)));

__asm("PWM1CON0 equ 016h");


typedef union {
    struct {
        unsigned :4;
        unsigned PWM1POL :1;
        unsigned PWM1OUT :1;
        unsigned PWM1OE :1;
        unsigned PWM1EN :1;
    };
} PWM1CONbits_t;
extern volatile PWM1CONbits_t PWM1CONbits __attribute__((address(0x016)));
# 882 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
typedef union {
    struct {
        unsigned :4;
        unsigned PWM1POL :1;
        unsigned PWM1OUT :1;
        unsigned PWM1OE :1;
        unsigned PWM1EN :1;
    };
} PWM1CON0bits_t;
extern volatile PWM1CON0bits_t PWM1CON0bits __attribute__((address(0x016)));
# 916 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PWM2DCL __attribute__((address(0x017)));

__asm("PWM2DCL equ 017h");


typedef union {
    struct {
        unsigned :6;
        unsigned PWM2DCL :2;
    };
    struct {
        unsigned :6;
        unsigned PWM2DCL0 :1;
        unsigned PWM2DCL1 :1;
    };
} PWM2DCLbits_t;
extern volatile PWM2DCLbits_t PWM2DCLbits __attribute__((address(0x017)));
# 952 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PWM2DCH __attribute__((address(0x018)));

__asm("PWM2DCH equ 018h");


typedef union {
    struct {
        unsigned PWM2DCH :8;
    };
    struct {
        unsigned PWM2DCH0 :1;
        unsigned PWM2DCH1 :1;
        unsigned PWM2DCH2 :1;
        unsigned PWM2DCH3 :1;
        unsigned PWM2DCH4 :1;
        unsigned PWM2DCH5 :1;
        unsigned PWM2DCH6 :1;
        unsigned PWM2DCH7 :1;
    };
} PWM2DCHbits_t;
extern volatile PWM2DCHbits_t PWM2DCHbits __attribute__((address(0x018)));
# 1022 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PWM2CON __attribute__((address(0x019)));

__asm("PWM2CON equ 019h");


extern volatile unsigned char PWM2CON0 __attribute__((address(0x019)));

__asm("PWM2CON0 equ 019h");


typedef union {
    struct {
        unsigned :4;
        unsigned PWM2POL :1;
        unsigned PWM2OUT :1;
        unsigned PWM2OE :1;
        unsigned PWM2EN :1;
    };
} PWM2CONbits_t;
extern volatile PWM2CONbits_t PWM2CONbits __attribute__((address(0x019)));
# 1064 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
typedef union {
    struct {
        unsigned :4;
        unsigned PWM2POL :1;
        unsigned PWM2OUT :1;
        unsigned PWM2OE :1;
        unsigned PWM2EN :1;
    };
} PWM2CON0bits_t;
extern volatile PWM2CON0bits_t PWM2CON0bits __attribute__((address(0x019)));
# 1098 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char IOCAP __attribute__((address(0x01A)));

__asm("IOCAP equ 01Ah");


typedef union {
    struct {
        unsigned IOCAP0 :1;
        unsigned IOCAP1 :1;
        unsigned IOCAP2 :1;
        unsigned IOCAP3 :1;
    };
} IOCAPbits_t;
extern volatile IOCAPbits_t IOCAPbits __attribute__((address(0x01A)));
# 1136 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char IOCAN __attribute__((address(0x01B)));

__asm("IOCAN equ 01Bh");


typedef union {
    struct {
        unsigned IOCAN0 :1;
        unsigned IOCAN1 :1;
        unsigned IOCAN2 :1;
        unsigned IOCAN3 :1;
    };
} IOCANbits_t;
extern volatile IOCANbits_t IOCANbits __attribute__((address(0x01B)));
# 1174 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char IOCAF __attribute__((address(0x01C)));

__asm("IOCAF equ 01Ch");


typedef union {
    struct {
        unsigned IOCAF0 :1;
        unsigned IOCAF1 :1;
        unsigned IOCAF2 :1;
        unsigned IOCAF3 :1;
    };
} IOCAFbits_t;
extern volatile IOCAFbits_t IOCAFbits __attribute__((address(0x01C)));
# 1212 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char FVRCON __attribute__((address(0x01D)));

__asm("FVRCON equ 01Dh");


typedef union {
    struct {
        unsigned ADFVR :2;
        unsigned :2;
        unsigned TSRNG :1;
        unsigned TSEN :1;
        unsigned FVRRDY :1;
        unsigned FVREN :1;
    };
    struct {
        unsigned ADFVR0 :1;
        unsigned ADFVR1 :1;
    };
} FVRCONbits_t;
extern volatile FVRCONbits_t FVRCONbits __attribute__((address(0x01D)));
# 1271 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char ADRES __attribute__((address(0x01E)));

__asm("ADRES equ 01Eh");




extern volatile unsigned char ADCON __attribute__((address(0x01F)));

__asm("ADCON equ 01Fh");


typedef union {
    struct {
        unsigned ADON :1;
        unsigned GO_nDONE :1;
        unsigned CHS :3;
        unsigned ADCS :3;
    };
    struct {
        unsigned :2;
        unsigned CHS0 :1;
        unsigned CHS1 :1;
        unsigned CHS2 :1;
        unsigned ADCS0 :1;
        unsigned ADCS1 :1;
        unsigned ADCS2 :1;
    };
} ADCONbits_t;
extern volatile ADCONbits_t ADCONbits __attribute__((address(0x01F)));
# 1355 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned short PMADR __attribute__((address(0x020)));

__asm("PMADR equ 020h");




extern volatile unsigned char PMADRL __attribute__((address(0x020)));

__asm("PMADRL equ 020h");


typedef union {
    struct {
        unsigned PMADR :8;
    };
} PMADRLbits_t;
extern volatile PMADRLbits_t PMADRLbits __attribute__((address(0x020)));
# 1382 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PMADRH __attribute__((address(0x021)));

__asm("PMADRH equ 021h");


typedef union {
    struct {
        unsigned PMADR8 :1;
    };
} PMADRHbits_t;
extern volatile PMADRHbits_t PMADRHbits __attribute__((address(0x021)));
# 1402 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned short PMDAT __attribute__((address(0x022)));

__asm("PMDAT equ 022h");




extern volatile unsigned char PMDATL __attribute__((address(0x022)));

__asm("PMDATL equ 022h");


typedef union {
    struct {
        unsigned PMDATL :8;
    };
} PMDATLbits_t;
extern volatile PMDATLbits_t PMDATLbits __attribute__((address(0x022)));
# 1429 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PMDATH __attribute__((address(0x023)));

__asm("PMDATH equ 023h");


typedef union {
    struct {
        unsigned PMDATH :6;
    };
} PMDATHbits_t;
extern volatile PMDATHbits_t PMDATHbits __attribute__((address(0x023)));
# 1449 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PMCON1 __attribute__((address(0x024)));

__asm("PMCON1 equ 024h");


typedef union {
    struct {
        unsigned RD :1;
        unsigned WR :1;
        unsigned WREN :1;
        unsigned WRERR :1;
        unsigned FREE :1;
        unsigned LWLO :1;
        unsigned CFGS :1;
    };
} PMCON1bits_t;
extern volatile PMCON1bits_t PMCON1bits __attribute__((address(0x024)));
# 1505 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char PMCON2 __attribute__((address(0x025)));

__asm("PMCON2 equ 025h");


typedef union {
    struct {
        unsigned PMCON2 :8;
    };
} PMCON2bits_t;
extern volatile PMCON2bits_t PMCON2bits __attribute__((address(0x025)));
# 1525 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CLKRCON __attribute__((address(0x026)));

__asm("CLKRCON equ 026h");


typedef union {
    struct {
        unsigned :6;
        unsigned CLKROE :1;
    };
} CLKRCONbits_t;
extern volatile CLKRCONbits_t CLKRCONbits __attribute__((address(0x026)));
# 1547 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile __uint24 NCO1ACC __attribute__((address(0x027)));


__asm("NCO1ACC equ 027h");




extern volatile unsigned char NCO1ACCL __attribute__((address(0x027)));

__asm("NCO1ACCL equ 027h");


typedef union {
    struct {
        unsigned NCO1ACC0 :1;
        unsigned NCO1ACC1 :1;
        unsigned NCO1ACC2 :1;
        unsigned NCO1ACC3 :1;
        unsigned NCO1ACC4 :1;
        unsigned NCO1ACC5 :1;
        unsigned NCO1ACC6 :1;
        unsigned NCO1ACC7 :1;
    };
} NCO1ACCLbits_t;
extern volatile NCO1ACCLbits_t NCO1ACCLbits __attribute__((address(0x027)));
# 1617 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char NCO1ACCH __attribute__((address(0x028)));

__asm("NCO1ACCH equ 028h");


typedef union {
    struct {
        unsigned NCO1ACC8 :1;
        unsigned NCO1ACC9 :1;
        unsigned NCO1ACC10 :1;
        unsigned NCO1ACC11 :1;
        unsigned NCO1ACC12 :1;
        unsigned NCO1ACC13 :1;
        unsigned NCO1ACC14 :1;
        unsigned NCO1ACC15 :1;
    };
} NCO1ACCHbits_t;
extern volatile NCO1ACCHbits_t NCO1ACCHbits __attribute__((address(0x028)));
# 1679 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char NCO1ACCU __attribute__((address(0x029)));

__asm("NCO1ACCU equ 029h");


typedef union {
    struct {
        unsigned NCO1ACC16 :1;
        unsigned NCO1ACC17 :1;
        unsigned NCO1ACC18 :1;
        unsigned NCO1ACC19 :1;
    };
} NCO1ACCUbits_t;
extern volatile NCO1ACCUbits_t NCO1ACCUbits __attribute__((address(0x029)));
# 1718 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile __uint24 NCO1INC __attribute__((address(0x02A)));


__asm("NCO1INC equ 02Ah");




extern volatile unsigned char NCO1INCL __attribute__((address(0x02A)));

__asm("NCO1INCL equ 02Ah");


typedef union {
    struct {
        unsigned NCO1INC0 :1;
        unsigned NCO1INC1 :1;
        unsigned NCO1INC2 :1;
        unsigned NCO1INC3 :1;
        unsigned NCO1INC4 :1;
        unsigned NCO1INC5 :1;
        unsigned NCO1INC6 :1;
        unsigned NCO1INC7 :1;
    };
} NCO1INCLbits_t;
extern volatile NCO1INCLbits_t NCO1INCLbits __attribute__((address(0x02A)));
# 1788 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char NCO1INCH __attribute__((address(0x02B)));

__asm("NCO1INCH equ 02Bh");


typedef union {
    struct {
        unsigned NCO1INC8 :1;
        unsigned NCO1INC9 :1;
        unsigned NCO1INC10 :1;
        unsigned NCO1INC11 :1;
        unsigned NCO1INC12 :1;
        unsigned NCO1INC13 :1;
        unsigned NCO1INC14 :1;
        unsigned NCO1INC15 :1;
    };
} NCO1INCHbits_t;
extern volatile NCO1INCHbits_t NCO1INCHbits __attribute__((address(0x02B)));
# 1850 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char NCO1INCU __attribute__((address(0x02C)));

__asm("NCO1INCU equ 02Ch");




extern volatile unsigned char NCO1CON __attribute__((address(0x02D)));

__asm("NCO1CON equ 02Dh");


typedef union {
    struct {
        unsigned N1PFM :1;
        unsigned :3;
        unsigned N1POL :1;
        unsigned N1OUT :1;
        unsigned N1OE :1;
        unsigned N1EN :1;
    };
} NCO1CONbits_t;
extern volatile NCO1CONbits_t NCO1CONbits __attribute__((address(0x02D)));
# 1902 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char NCO1CLK __attribute__((address(0x02E)));

__asm("NCO1CLK equ 02Eh");


typedef union {
    struct {
        unsigned N1CKS :4;
        unsigned :1;
        unsigned N1PWS :3;
    };
    struct {
        unsigned N1CKS0 :1;
        unsigned N1CKS1 :1;
        unsigned :3;
        unsigned N1PWS0 :1;
        unsigned N1PWS1 :1;
        unsigned N1PWS2 :1;
    };
} NCO1CLKbits_t;
extern volatile NCO1CLKbits_t NCO1CLKbits __attribute__((address(0x02E)));
# 1962 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char WDTCON __attribute__((address(0x030)));

__asm("WDTCON equ 030h");


typedef union {
    struct {
        unsigned SWDTEN :1;
        unsigned WDTPS :5;
    };
    struct {
        unsigned :1;
        unsigned WDTPS0 :1;
        unsigned WDTPS1 :1;
        unsigned WDTPS2 :1;
        unsigned WDTPS3 :1;
        unsigned WDTPS4 :1;
    };
} WDTCONbits_t;
extern volatile WDTCONbits_t WDTCONbits __attribute__((address(0x030)));
# 2021 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CLC1CON __attribute__((address(0x031)));

__asm("CLC1CON equ 031h");


typedef union {
    struct {
        unsigned LC1MODE0 :1;
        unsigned LC1MODE1 :1;
        unsigned LC1MODE2 :1;
        unsigned LC1INTN :1;
        unsigned LC1INTP :1;
        unsigned LC1OUT :1;
        unsigned LC1OE :1;
        unsigned LC1EN :1;
    };
    struct {
        unsigned LCMODE0 :1;
        unsigned LCMODE1 :1;
        unsigned LCMODE2 :1;
        unsigned LCINTN :1;
        unsigned LCINTP :1;
        unsigned LCOUT :1;
        unsigned LCOE :1;
        unsigned LCEN :1;
    };
    struct {
        unsigned LC1MODE :3;
    };
} CLC1CONbits_t;
extern volatile CLC1CONbits_t CLC1CONbits __attribute__((address(0x031)));
# 2141 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CLC1SEL0 __attribute__((address(0x032)));

__asm("CLC1SEL0 equ 032h");


typedef union {
    struct {
        unsigned LC1D1S0 :1;
        unsigned LC1D1S1 :1;
        unsigned LC1D1S2 :1;
        unsigned :1;
        unsigned LC1D2S0 :1;
        unsigned LC1D2S1 :1;
        unsigned LC1D2S2 :1;
    };
    struct {
        unsigned D1S0 :1;
        unsigned D1S1 :1;
        unsigned D1S2 :1;
        unsigned :1;
        unsigned D2S0 :1;
        unsigned D2S1 :1;
        unsigned D2S2 :1;
    };
    struct {
        unsigned LC1D1S :3;
        unsigned :1;
        unsigned LC1D2S :3;
    };
} CLC1SEL0bits_t;
extern volatile CLC1SEL0bits_t CLC1SEL0bits __attribute__((address(0x032)));
# 2246 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CLC1SEL1 __attribute__((address(0x033)));

__asm("CLC1SEL1 equ 033h");


typedef union {
    struct {
        unsigned LC1D3S0 :1;
        unsigned LC1D3S1 :1;
        unsigned LC1D3S2 :1;
        unsigned :1;
        unsigned LC1D4S0 :1;
        unsigned LC1D4S1 :1;
        unsigned LC1D4S2 :1;
    };
    struct {
        unsigned D3S0 :1;
        unsigned D3S1 :1;
        unsigned D3S2 :1;
        unsigned :1;
        unsigned D4S0 :1;
        unsigned D4S1 :1;
        unsigned D4S2 :1;
    };
    struct {
        unsigned LC1D3S :3;
        unsigned :1;
        unsigned LC1D4S :3;
    };
} CLC1SEL1bits_t;
extern volatile CLC1SEL1bits_t CLC1SEL1bits __attribute__((address(0x033)));
# 2351 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CLC1POL __attribute__((address(0x034)));

__asm("CLC1POL equ 034h");


typedef union {
    struct {
        unsigned LC1G1POL :1;
        unsigned LC1G2POL :1;
        unsigned LC1G3POL :1;
        unsigned LC1G4POL :1;
        unsigned :3;
        unsigned LC1POL :1;
    };
    struct {
        unsigned G1POL :1;
        unsigned G2POL :1;
        unsigned G3POL :1;
        unsigned G4POL :1;
        unsigned :3;
        unsigned POL :1;
    };
} CLC1POLbits_t;
extern volatile CLC1POLbits_t CLC1POLbits __attribute__((address(0x034)));
# 2429 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CLC1GLS0 __attribute__((address(0x035)));

__asm("CLC1GLS0 equ 035h");


typedef union {
    struct {
        unsigned LC1G1D1N :1;
        unsigned LC1G1D1T :1;
        unsigned LC1G1D2N :1;
        unsigned LC1G1D2T :1;
        unsigned LC1G1D3N :1;
        unsigned LC1G1D3T :1;
        unsigned LC1G1D4N :1;
        unsigned LC1G1D4T :1;
    };
    struct {
        unsigned D1N :1;
        unsigned D1T :1;
        unsigned D2N :1;
        unsigned D2T :1;
        unsigned D3N :1;
        unsigned D3T :1;
        unsigned D4N :1;
        unsigned D4T :1;
    };
} CLC1GLS0bits_t;
extern volatile CLC1GLS0bits_t CLC1GLS0bits __attribute__((address(0x035)));
# 2541 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CLC1GLS1 __attribute__((address(0x036)));

__asm("CLC1GLS1 equ 036h");


typedef union {
    struct {
        unsigned LC1G2D1N :1;
        unsigned LC1G2D1T :1;
        unsigned LC1G2D2N :1;
        unsigned LC1G2D2T :1;
        unsigned LC1G2D3N :1;
        unsigned LC1G2D3T :1;
        unsigned LC1G2D4N :1;
        unsigned LC1G2D4T :1;
    };
    struct {
        unsigned D1N :1;
        unsigned D1T :1;
        unsigned D2N :1;
        unsigned D2T :1;
        unsigned D3N :1;
        unsigned D3T :1;
        unsigned D4N :1;
        unsigned D4T :1;
    };
} CLC1GLS1bits_t;
extern volatile CLC1GLS1bits_t CLC1GLS1bits __attribute__((address(0x036)));
# 2653 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CLC1GLS2 __attribute__((address(0x037)));

__asm("CLC1GLS2 equ 037h");


typedef union {
    struct {
        unsigned LC1G3D1N :1;
        unsigned LC1G3D1T :1;
        unsigned LC1G3D2N :1;
        unsigned LC1G3D2T :1;
        unsigned LC1G3D3N :1;
        unsigned LC1G3D3T :1;
        unsigned LC1G3D4N :1;
        unsigned LC1G3D4T :1;
    };
    struct {
        unsigned D1N :1;
        unsigned D1T :1;
        unsigned D2N :1;
        unsigned D2T :1;
        unsigned D3N :1;
        unsigned D3T :1;
        unsigned D4N :1;
        unsigned D4T :1;
    };
} CLC1GLS2bits_t;
extern volatile CLC1GLS2bits_t CLC1GLS2bits __attribute__((address(0x037)));
# 2765 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CLC1GLS3 __attribute__((address(0x038)));

__asm("CLC1GLS3 equ 038h");


typedef union {
    struct {
        unsigned LC1G4D1N :1;
        unsigned LC1G4D1T :1;
        unsigned LC1G4D2N :1;
        unsigned LC1G4D2T :1;
        unsigned LC1G4D3N :1;
        unsigned LC1G4D3T :1;
        unsigned LC1G4D4N :1;
        unsigned LC1G4D4T :1;
    };
    struct {
        unsigned G4D1N :1;
        unsigned G4D1T :1;
        unsigned G4D2N :1;
        unsigned G4D2T :1;
        unsigned G4D3N :1;
        unsigned G4D3T :1;
        unsigned G4D4N :1;
        unsigned G4D4T :1;
    };
} CLC1GLS3bits_t;
extern volatile CLC1GLS3bits_t CLC1GLS3bits __attribute__((address(0x038)));
# 2877 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CWG1CON0 __attribute__((address(0x039)));

__asm("CWG1CON0 equ 039h");


typedef union {
    struct {
        unsigned G1CS0 :1;
        unsigned :2;
        unsigned G1POLA :1;
        unsigned G1POLB :1;
        unsigned G1OEA :1;
        unsigned G1OEB :1;
        unsigned G1EN :1;
    };
    struct {
        unsigned G1CS :2;
    };
} CWG1CON0bits_t;
extern volatile CWG1CON0bits_t CWG1CON0bits __attribute__((address(0x039)));
# 2936 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CWG1CON1 __attribute__((address(0x03A)));

__asm("CWG1CON1 equ 03Ah");


typedef union {
    struct {
        unsigned G1IS0 :1;
        unsigned G1IS1 :1;
        unsigned :2;
        unsigned G1ASDLA :2;
        unsigned G1ASDLB :2;
    };
    struct {
        unsigned G1IS :4;
        unsigned G1ASDLA0 :1;
        unsigned G1ASDLA1 :1;
        unsigned G1ASDLB0 :1;
        unsigned G1ASDLB1 :1;
    };
} CWG1CON1bits_t;
extern volatile CWG1CON1bits_t CWG1CON1bits __attribute__((address(0x03A)));
# 3007 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CWG1CON2 __attribute__((address(0x03B)));

__asm("CWG1CON2 equ 03Bh");


typedef union {
    struct {
        unsigned G1ASDSFLT :1;
        unsigned G1ASDSCLC1 :1;
        unsigned :4;
        unsigned G1ARSEN :1;
        unsigned G1ASE :1;
    };
} CWG1CON2bits_t;
extern volatile CWG1CON2bits_t CWG1CON2bits __attribute__((address(0x03B)));
# 3046 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CWG1DBR __attribute__((address(0x03C)));

__asm("CWG1DBR equ 03Ch");


typedef union {
    struct {
        unsigned CWG1DBR :6;
    };
    struct {
        unsigned CWG1DBR0 :1;
        unsigned CWG1DBR1 :1;
        unsigned CWG1DBR2 :1;
        unsigned CWG1DBR3 :1;
        unsigned CWG1DBR4 :1;
        unsigned CWG1DBR5 :1;
    };
} CWG1DBRbits_t;
extern volatile CWG1DBRbits_t CWG1DBRbits __attribute__((address(0x03C)));
# 3104 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char CWG1DBF __attribute__((address(0x03D)));

__asm("CWG1DBF equ 03Dh");


typedef union {
    struct {
        unsigned CWG1DBF :6;
    };
    struct {
        unsigned CWG1DBF0 :1;
        unsigned CWG1DBF1 :1;
        unsigned CWG1DBF2 :1;
        unsigned CWG1DBF3 :1;
        unsigned CWG1DBF4 :1;
        unsigned CWG1DBF5 :1;
    };
} CWG1DBFbits_t;
extern volatile CWG1DBFbits_t CWG1DBFbits __attribute__((address(0x03D)));
# 3162 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char VREGCON __attribute__((address(0x03E)));

__asm("VREGCON equ 03Eh");


typedef union {
    struct {
        unsigned VREGPM :2;
    };
    struct {
        unsigned VREGPM0 :1;
        unsigned VREGPM1 :1;
    };
} VREGCONbits_t;
extern volatile VREGCONbits_t VREGCONbits __attribute__((address(0x03E)));
# 3196 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile unsigned char BORCON __attribute__((address(0x03F)));

__asm("BORCON equ 03Fh");


typedef union {
    struct {
        unsigned BORRDY :1;
        unsigned :5;
        unsigned BORFS :1;
        unsigned SBOREN :1;
    };
} BORCONbits_t;
extern volatile BORCONbits_t BORCONbits __attribute__((address(0x03F)));
# 3235 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\proc\\pic10lf322.h" 3
extern volatile __bit ADCS0 __attribute__((address(0xFD)));


extern volatile __bit ADCS1 __attribute__((address(0xFE)));


extern volatile __bit ADCS2 __attribute__((address(0xFF)));


extern volatile __bit ADFVR0 __attribute__((address(0xE8)));


extern volatile __bit ADFVR1 __attribute__((address(0xE9)));


extern volatile __bit ADIE __attribute__((address(0x6E)));


extern volatile __bit ADIF __attribute__((address(0x66)));


extern volatile __bit ADON __attribute__((address(0xF8)));


extern volatile __bit ANSA0 __attribute__((address(0x40)));


extern volatile __bit ANSA1 __attribute__((address(0x41)));


extern volatile __bit ANSA2 __attribute__((address(0x42)));


extern volatile __bit BORFS __attribute__((address(0x1FE)));


extern volatile __bit BORRDY __attribute__((address(0x1F8)));


extern volatile __bit CARRY __attribute__((address(0x18)));


extern volatile __bit CFGS __attribute__((address(0x126)));


extern volatile __bit CHS0 __attribute__((address(0xFA)));


extern volatile __bit CHS1 __attribute__((address(0xFB)));


extern volatile __bit CHS2 __attribute__((address(0xFC)));


extern volatile __bit CLC1IE __attribute__((address(0x6B)));


extern volatile __bit CLC1IF __attribute__((address(0x63)));


extern volatile __bit CLKROE __attribute__((address(0x136)));


extern volatile __bit CWG1DBF0 __attribute__((address(0x1E8)));


extern volatile __bit CWG1DBF1 __attribute__((address(0x1E9)));


extern volatile __bit CWG1DBF2 __attribute__((address(0x1EA)));


extern volatile __bit CWG1DBF3 __attribute__((address(0x1EB)));


extern volatile __bit CWG1DBF4 __attribute__((address(0x1EC)));


extern volatile __bit CWG1DBF5 __attribute__((address(0x1ED)));


extern volatile __bit CWG1DBR0 __attribute__((address(0x1E0)));


extern volatile __bit CWG1DBR1 __attribute__((address(0x1E1)));


extern volatile __bit CWG1DBR2 __attribute__((address(0x1E2)));


extern volatile __bit CWG1DBR3 __attribute__((address(0x1E3)));


extern volatile __bit CWG1DBR4 __attribute__((address(0x1E4)));


extern volatile __bit CWG1DBR5 __attribute__((address(0x1E5)));


extern volatile __bit D1S0 __attribute__((address(0x190)));


extern volatile __bit D1S1 __attribute__((address(0x191)));


extern volatile __bit D1S2 __attribute__((address(0x192)));


extern volatile __bit D2S0 __attribute__((address(0x194)));


extern volatile __bit D2S1 __attribute__((address(0x195)));


extern volatile __bit D2S2 __attribute__((address(0x196)));


extern volatile __bit D3S0 __attribute__((address(0x198)));


extern volatile __bit D3S1 __attribute__((address(0x199)));


extern volatile __bit D3S2 __attribute__((address(0x19A)));


extern volatile __bit D4S0 __attribute__((address(0x19C)));


extern volatile __bit D4S1 __attribute__((address(0x19D)));


extern volatile __bit D4S2 __attribute__((address(0x19E)));


extern volatile __bit DC __attribute__((address(0x19)));


extern volatile __bit FREE __attribute__((address(0x124)));


extern volatile __bit FVREN __attribute__((address(0xEF)));


extern volatile __bit FVRRDY __attribute__((address(0xEE)));


extern volatile __bit G1ARSEN __attribute__((address(0x1DE)));


extern volatile __bit G1ASDLA0 __attribute__((address(0x1D4)));


extern volatile __bit G1ASDLA1 __attribute__((address(0x1D5)));


extern volatile __bit G1ASDLB0 __attribute__((address(0x1D6)));


extern volatile __bit G1ASDLB1 __attribute__((address(0x1D7)));


extern volatile __bit G1ASDSCLC1 __attribute__((address(0x1D9)));


extern volatile __bit G1ASDSFLT __attribute__((address(0x1D8)));


extern volatile __bit G1ASE __attribute__((address(0x1DF)));


extern volatile __bit G1CS0 __attribute__((address(0x1C8)));


extern volatile __bit G1EN __attribute__((address(0x1CF)));


extern volatile __bit G1IS0 __attribute__((address(0x1D0)));


extern volatile __bit G1IS1 __attribute__((address(0x1D1)));


extern volatile __bit G1OEA __attribute__((address(0x1CD)));


extern volatile __bit G1OEB __attribute__((address(0x1CE)));


extern volatile __bit G1POL __attribute__((address(0x1A0)));


extern volatile __bit G1POLA __attribute__((address(0x1CB)));


extern volatile __bit G1POLB __attribute__((address(0x1CC)));


extern volatile __bit G2POL __attribute__((address(0x1A1)));


extern volatile __bit G3POL __attribute__((address(0x1A2)));


extern volatile __bit G4D1N __attribute__((address(0x1C0)));


extern volatile __bit G4D1T __attribute__((address(0x1C1)));


extern volatile __bit G4D2N __attribute__((address(0x1C2)));


extern volatile __bit G4D2T __attribute__((address(0x1C3)));


extern volatile __bit G4D3N __attribute__((address(0x1C4)));


extern volatile __bit G4D3T __attribute__((address(0x1C5)));


extern volatile __bit G4D4N __attribute__((address(0x1C6)));


extern volatile __bit G4D4T __attribute__((address(0x1C7)));


extern volatile __bit G4POL __attribute__((address(0x1A3)));


extern volatile __bit GIE __attribute__((address(0x5F)));


extern volatile __bit GO_nDONE __attribute__((address(0xF9)));


extern volatile __bit HFIOFR __attribute__((address(0x83)));


extern volatile __bit HFIOFS __attribute__((address(0x80)));


extern volatile __bit INTE __attribute__((address(0x5C)));


extern volatile __bit INTEDG __attribute__((address(0x76)));


extern volatile __bit INTF __attribute__((address(0x59)));


extern volatile __bit IOCAF0 __attribute__((address(0xE0)));


extern volatile __bit IOCAF1 __attribute__((address(0xE1)));


extern volatile __bit IOCAF2 __attribute__((address(0xE2)));


extern volatile __bit IOCAF3 __attribute__((address(0xE3)));


extern volatile __bit IOCAN0 __attribute__((address(0xD8)));


extern volatile __bit IOCAN1 __attribute__((address(0xD9)));


extern volatile __bit IOCAN2 __attribute__((address(0xDA)));


extern volatile __bit IOCAN3 __attribute__((address(0xDB)));


extern volatile __bit IOCAP0 __attribute__((address(0xD0)));


extern volatile __bit IOCAP1 __attribute__((address(0xD1)));


extern volatile __bit IOCAP2 __attribute__((address(0xD2)));


extern volatile __bit IOCAP3 __attribute__((address(0xD3)));


extern volatile __bit IOCIE __attribute__((address(0x5B)));


extern volatile __bit IOCIF __attribute__((address(0x58)));


extern volatile __bit IRCF0 __attribute__((address(0x84)));


extern volatile __bit IRCF1 __attribute__((address(0x85)));


extern volatile __bit IRCF2 __attribute__((address(0x86)));


extern volatile __bit IRP __attribute__((address(0x1F)));


extern volatile __bit LATA0 __attribute__((address(0x38)));


extern volatile __bit LATA1 __attribute__((address(0x39)));


extern volatile __bit LATA2 __attribute__((address(0x3A)));


extern volatile __bit LC1D1S0 __attribute__((address(0x190)));


extern volatile __bit LC1D1S1 __attribute__((address(0x191)));


extern volatile __bit LC1D1S2 __attribute__((address(0x192)));


extern volatile __bit LC1D2S0 __attribute__((address(0x194)));


extern volatile __bit LC1D2S1 __attribute__((address(0x195)));


extern volatile __bit LC1D2S2 __attribute__((address(0x196)));


extern volatile __bit LC1D3S0 __attribute__((address(0x198)));


extern volatile __bit LC1D3S1 __attribute__((address(0x199)));


extern volatile __bit LC1D3S2 __attribute__((address(0x19A)));


extern volatile __bit LC1D4S0 __attribute__((address(0x19C)));


extern volatile __bit LC1D4S1 __attribute__((address(0x19D)));


extern volatile __bit LC1D4S2 __attribute__((address(0x19E)));


extern volatile __bit LC1EN __attribute__((address(0x18F)));


extern volatile __bit LC1G1D1N __attribute__((address(0x1A8)));


extern volatile __bit LC1G1D1T __attribute__((address(0x1A9)));


extern volatile __bit LC1G1D2N __attribute__((address(0x1AA)));


extern volatile __bit LC1G1D2T __attribute__((address(0x1AB)));


extern volatile __bit LC1G1D3N __attribute__((address(0x1AC)));


extern volatile __bit LC1G1D3T __attribute__((address(0x1AD)));


extern volatile __bit LC1G1D4N __attribute__((address(0x1AE)));


extern volatile __bit LC1G1D4T __attribute__((address(0x1AF)));


extern volatile __bit LC1G1POL __attribute__((address(0x1A0)));


extern volatile __bit LC1G2D1N __attribute__((address(0x1B0)));


extern volatile __bit LC1G2D1T __attribute__((address(0x1B1)));


extern volatile __bit LC1G2D2N __attribute__((address(0x1B2)));


extern volatile __bit LC1G2D2T __attribute__((address(0x1B3)));


extern volatile __bit LC1G2D3N __attribute__((address(0x1B4)));


extern volatile __bit LC1G2D3T __attribute__((address(0x1B5)));


extern volatile __bit LC1G2D4N __attribute__((address(0x1B6)));


extern volatile __bit LC1G2D4T __attribute__((address(0x1B7)));


extern volatile __bit LC1G2POL __attribute__((address(0x1A1)));


extern volatile __bit LC1G3D1N __attribute__((address(0x1B8)));


extern volatile __bit LC1G3D1T __attribute__((address(0x1B9)));


extern volatile __bit LC1G3D2N __attribute__((address(0x1BA)));


extern volatile __bit LC1G3D2T __attribute__((address(0x1BB)));


extern volatile __bit LC1G3D3N __attribute__((address(0x1BC)));


extern volatile __bit LC1G3D3T __attribute__((address(0x1BD)));


extern volatile __bit LC1G3D4N __attribute__((address(0x1BE)));


extern volatile __bit LC1G3D4T __attribute__((address(0x1BF)));


extern volatile __bit LC1G3POL __attribute__((address(0x1A2)));


extern volatile __bit LC1G4D1N __attribute__((address(0x1C0)));


extern volatile __bit LC1G4D1T __attribute__((address(0x1C1)));


extern volatile __bit LC1G4D2N __attribute__((address(0x1C2)));


extern volatile __bit LC1G4D2T __attribute__((address(0x1C3)));


extern volatile __bit LC1G4D3N __attribute__((address(0x1C4)));


extern volatile __bit LC1G4D3T __attribute__((address(0x1C5)));


extern volatile __bit LC1G4D4N __attribute__((address(0x1C6)));


extern volatile __bit LC1G4D4T __attribute__((address(0x1C7)));


extern volatile __bit LC1G4POL __attribute__((address(0x1A3)));


extern volatile __bit LC1INTN __attribute__((address(0x18B)));


extern volatile __bit LC1INTP __attribute__((address(0x18C)));


extern volatile __bit LC1MODE0 __attribute__((address(0x188)));


extern volatile __bit LC1MODE1 __attribute__((address(0x189)));


extern volatile __bit LC1MODE2 __attribute__((address(0x18A)));


extern volatile __bit LC1OE __attribute__((address(0x18E)));


extern volatile __bit LC1OUT __attribute__((address(0x18D)));


extern volatile __bit LC1POL __attribute__((address(0x1A7)));


extern volatile __bit LCEN __attribute__((address(0x18F)));


extern volatile __bit LCINTN __attribute__((address(0x18B)));


extern volatile __bit LCINTP __attribute__((address(0x18C)));


extern volatile __bit LCMODE0 __attribute__((address(0x188)));


extern volatile __bit LCMODE1 __attribute__((address(0x189)));


extern volatile __bit LCMODE2 __attribute__((address(0x18A)));


extern volatile __bit LCOE __attribute__((address(0x18E)));


extern volatile __bit LCOUT __attribute__((address(0x18D)));


extern volatile __bit LFIOFR __attribute__((address(0x81)));


extern volatile __bit LWLO __attribute__((address(0x125)));


extern volatile __bit N1CKS0 __attribute__((address(0x170)));


extern volatile __bit N1CKS1 __attribute__((address(0x171)));


extern volatile __bit N1EN __attribute__((address(0x16F)));


extern volatile __bit N1OE __attribute__((address(0x16E)));


extern volatile __bit N1OUT __attribute__((address(0x16D)));


extern volatile __bit N1PFM __attribute__((address(0x168)));


extern volatile __bit N1POL __attribute__((address(0x16C)));


extern volatile __bit N1PWS0 __attribute__((address(0x175)));


extern volatile __bit N1PWS1 __attribute__((address(0x176)));


extern volatile __bit N1PWS2 __attribute__((address(0x177)));


extern volatile __bit NCO1ACC0 __attribute__((address(0x138)));


extern volatile __bit NCO1ACC1 __attribute__((address(0x139)));


extern volatile __bit NCO1ACC10 __attribute__((address(0x142)));


extern volatile __bit NCO1ACC11 __attribute__((address(0x143)));


extern volatile __bit NCO1ACC12 __attribute__((address(0x144)));


extern volatile __bit NCO1ACC13 __attribute__((address(0x145)));


extern volatile __bit NCO1ACC14 __attribute__((address(0x146)));


extern volatile __bit NCO1ACC15 __attribute__((address(0x147)));


extern volatile __bit NCO1ACC16 __attribute__((address(0x148)));


extern volatile __bit NCO1ACC17 __attribute__((address(0x149)));


extern volatile __bit NCO1ACC18 __attribute__((address(0x14A)));


extern volatile __bit NCO1ACC19 __attribute__((address(0x14B)));


extern volatile __bit NCO1ACC2 __attribute__((address(0x13A)));


extern volatile __bit NCO1ACC3 __attribute__((address(0x13B)));


extern volatile __bit NCO1ACC4 __attribute__((address(0x13C)));


extern volatile __bit NCO1ACC5 __attribute__((address(0x13D)));


extern volatile __bit NCO1ACC6 __attribute__((address(0x13E)));


extern volatile __bit NCO1ACC7 __attribute__((address(0x13F)));


extern volatile __bit NCO1ACC8 __attribute__((address(0x140)));


extern volatile __bit NCO1ACC9 __attribute__((address(0x141)));


extern volatile __bit NCO1IE __attribute__((address(0x6C)));


extern volatile __bit NCO1IF __attribute__((address(0x64)));


extern volatile __bit NCO1INC0 __attribute__((address(0x150)));


extern volatile __bit NCO1INC1 __attribute__((address(0x151)));


extern volatile __bit NCO1INC10 __attribute__((address(0x15A)));


extern volatile __bit NCO1INC11 __attribute__((address(0x15B)));


extern volatile __bit NCO1INC12 __attribute__((address(0x15C)));


extern volatile __bit NCO1INC13 __attribute__((address(0x15D)));


extern volatile __bit NCO1INC14 __attribute__((address(0x15E)));


extern volatile __bit NCO1INC15 __attribute__((address(0x15F)));


extern volatile __bit NCO1INC2 __attribute__((address(0x152)));


extern volatile __bit NCO1INC3 __attribute__((address(0x153)));


extern volatile __bit NCO1INC4 __attribute__((address(0x154)));


extern volatile __bit NCO1INC5 __attribute__((address(0x155)));


extern volatile __bit NCO1INC6 __attribute__((address(0x156)));


extern volatile __bit NCO1INC7 __attribute__((address(0x157)));


extern volatile __bit NCO1INC8 __attribute__((address(0x158)));


extern volatile __bit NCO1INC9 __attribute__((address(0x159)));


extern volatile __bit PCLH0 __attribute__((address(0x50)));


extern volatile __bit PEIE __attribute__((address(0x5E)));


extern volatile __bit PMADR8 __attribute__((address(0x108)));


extern volatile __bit POL __attribute__((address(0x1A7)));


extern volatile __bit PS0 __attribute__((address(0x70)));


extern volatile __bit PS1 __attribute__((address(0x71)));


extern volatile __bit PS2 __attribute__((address(0x72)));


extern volatile __bit PSA __attribute__((address(0x73)));


extern volatile __bit PWM1DCH0 __attribute__((address(0xA8)));


extern volatile __bit PWM1DCH1 __attribute__((address(0xA9)));


extern volatile __bit PWM1DCH2 __attribute__((address(0xAA)));


extern volatile __bit PWM1DCH3 __attribute__((address(0xAB)));


extern volatile __bit PWM1DCH4 __attribute__((address(0xAC)));


extern volatile __bit PWM1DCH5 __attribute__((address(0xAD)));


extern volatile __bit PWM1DCH6 __attribute__((address(0xAE)));


extern volatile __bit PWM1DCH7 __attribute__((address(0xAF)));


extern volatile __bit PWM1DCL0 __attribute__((address(0xA6)));


extern volatile __bit PWM1DCL1 __attribute__((address(0xA7)));


extern volatile __bit PWM1EN __attribute__((address(0xB7)));


extern volatile __bit PWM1OE __attribute__((address(0xB6)));


extern volatile __bit PWM1OUT __attribute__((address(0xB5)));


extern volatile __bit PWM1POL __attribute__((address(0xB4)));


extern volatile __bit PWM2DCH0 __attribute__((address(0xC0)));


extern volatile __bit PWM2DCH1 __attribute__((address(0xC1)));


extern volatile __bit PWM2DCH2 __attribute__((address(0xC2)));


extern volatile __bit PWM2DCH3 __attribute__((address(0xC3)));


extern volatile __bit PWM2DCH4 __attribute__((address(0xC4)));


extern volatile __bit PWM2DCH5 __attribute__((address(0xC5)));


extern volatile __bit PWM2DCH6 __attribute__((address(0xC6)));


extern volatile __bit PWM2DCH7 __attribute__((address(0xC7)));


extern volatile __bit PWM2DCL0 __attribute__((address(0xBE)));


extern volatile __bit PWM2DCL1 __attribute__((address(0xBF)));


extern volatile __bit PWM2EN __attribute__((address(0xCF)));


extern volatile __bit PWM2OE __attribute__((address(0xCE)));


extern volatile __bit PWM2OUT __attribute__((address(0xCD)));


extern volatile __bit PWM2POL __attribute__((address(0xCC)));


extern volatile __bit RA0 __attribute__((address(0x28)));


extern volatile __bit RA1 __attribute__((address(0x29)));


extern volatile __bit RA2 __attribute__((address(0x2A)));


extern volatile __bit RA3 __attribute__((address(0x2B)));


extern volatile __bit RD __attribute__((address(0x120)));


extern volatile __bit RP0 __attribute__((address(0x1D)));


extern volatile __bit RP1 __attribute__((address(0x1E)));


extern volatile __bit SBOREN __attribute__((address(0x1FF)));


extern volatile __bit SWDTEN __attribute__((address(0x180)));


extern volatile __bit T0CS __attribute__((address(0x75)));


extern volatile __bit T0SE __attribute__((address(0x74)));


extern volatile __bit T2CKPS0 __attribute__((address(0x98)));


extern volatile __bit T2CKPS1 __attribute__((address(0x99)));


extern volatile __bit TMR0IE __attribute__((address(0x5D)));


extern volatile __bit TMR0IF __attribute__((address(0x5A)));


extern volatile __bit TMR2IE __attribute__((address(0x69)));


extern volatile __bit TMR2IF __attribute__((address(0x61)));


extern volatile __bit TMR2ON __attribute__((address(0x9A)));


extern volatile __bit TOUTPS0 __attribute__((address(0x9B)));


extern volatile __bit TOUTPS1 __attribute__((address(0x9C)));


extern volatile __bit TOUTPS2 __attribute__((address(0x9D)));


extern volatile __bit TOUTPS3 __attribute__((address(0x9E)));


extern volatile __bit TRISA0 __attribute__((address(0x30)));


extern volatile __bit TRISA1 __attribute__((address(0x31)));


extern volatile __bit TRISA2 __attribute__((address(0x32)));


extern volatile __bit TSEN __attribute__((address(0xED)));


extern volatile __bit TSRNG __attribute__((address(0xEC)));


extern volatile __bit VREGPM0 __attribute__((address(0x1F0)));


extern volatile __bit VREGPM1 __attribute__((address(0x1F1)));


extern volatile __bit WDTPS0 __attribute__((address(0x181)));


extern volatile __bit WDTPS1 __attribute__((address(0x182)));


extern volatile __bit WDTPS2 __attribute__((address(0x183)));


extern volatile __bit WDTPS3 __attribute__((address(0x184)));


extern volatile __bit WDTPS4 __attribute__((address(0x185)));


extern volatile __bit WPUA0 __attribute__((address(0x48)));


extern volatile __bit WPUA1 __attribute__((address(0x49)));


extern volatile __bit WPUA2 __attribute__((address(0x4A)));


extern volatile __bit WPUA3 __attribute__((address(0x4B)));


extern volatile __bit WR __attribute__((address(0x121)));


extern volatile __bit WREN __attribute__((address(0x122)));


extern volatile __bit WRERR __attribute__((address(0x123)));


extern volatile __bit ZERO __attribute__((address(0x1A)));


extern volatile __bit nBOR __attribute__((address(0x78)));


extern volatile __bit nPD __attribute__((address(0x1B)));


extern volatile __bit nPOR __attribute__((address(0x79)));


extern volatile __bit nTO __attribute__((address(0x1C)));


extern volatile __bit nWPUEN __attribute__((address(0x77)));
# 115 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\pic_chip_select.h" 2 3
# 14 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\pic.h" 2 3
# 30 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\pic.h" 3
#pragma intrinsic(__nop)
extern void __nop(void);
# 78 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\pic.h" 3
__attribute__((__unsupported__("The " "FLASH_READ" " macro function is no longer supported. Please use the MPLAB X MCC."))) unsigned char __flash_read(unsigned short addr);

__attribute__((__unsupported__("The " "FLASH_WRITE" " macro function is no longer supported. Please use the MPLAB X MCC."))) void __flash_write(unsigned short addr, unsigned short data);

__attribute__((__unsupported__("The " "FLASH_ERASE" " macro function is no longer supported. Please use the MPLAB X MCC."))) void __flash_erase(unsigned short addr);



# 1 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\eeprom_routines.h" 1 3
# 86 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\pic.h" 2 3





#pragma intrinsic(_delay)
extern __attribute__((nonreentrant)) void _delay(unsigned long);
#pragma intrinsic(_delaywdt)
extern __attribute__((nonreentrant)) void _delaywdt(unsigned long);
# 137 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\pic.h" 3
extern __bank0 unsigned char __resetbits;
extern __bank0 __bit __powerdown;
extern __bank0 __bit __timeout;
# 28 "C:/Program Files (x86)/Microchip/MPLABX/v5.40/packs/Microchip/PIC10-12Fxxx_DFP/1.3.46/xc8\\pic\\include\\xc.h" 2 3
# 54 "mcc_generated_files/pin_manager.h" 2
# 136 "mcc_generated_files/pin_manager.h"
void PIN_MANAGER_Initialize (void);
# 49 "mcc_generated_files/pin_manager.c" 2

# 1 "mcc_generated_files/../header.h" 1
# 10 "mcc_generated_files/../header.h"
# 1 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 1 3
# 13 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef signed char int8_t;






typedef signed int int16_t;







typedef __int24 int24_t;







typedef signed long int int32_t;
# 52 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef unsigned char uint8_t;





typedef unsigned int uint16_t;






typedef __uint24 uint24_t;






typedef unsigned long int uint32_t;
# 88 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef signed char int_least8_t;







typedef signed int int_least16_t;
# 109 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef __int24 int_least24_t;
# 118 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef signed long int int_least32_t;
# 136 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef unsigned char uint_least8_t;






typedef unsigned int uint_least16_t;
# 154 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef __uint24 uint_least24_t;







typedef unsigned long int uint_least32_t;
# 181 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef signed char int_fast8_t;






typedef signed int int_fast16_t;
# 200 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef __int24 int_fast24_t;







typedef signed long int int_fast32_t;
# 224 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef unsigned char uint_fast8_t;





typedef unsigned int uint_fast16_t;
# 240 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef __uint24 uint_fast24_t;






typedef unsigned long int uint_fast32_t;
# 268 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef int32_t intmax_t;
# 282 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 3
typedef uint32_t uintmax_t;






typedef int16_t intptr_t;




typedef uint16_t uintptr_t;
# 10 "mcc_generated_files/../header.h" 2

# 1 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdbool.h" 1 3
# 11 "mcc_generated_files/../header.h" 2
# 27 "mcc_generated_files/../header.h"
long counter;

struct DICE {
   unsigned int next : 4;
   unsigned int current : 4;
};

struct DICE dice;

char numbers[7] = {0b0000, 0b0100, 0b0010, 0b0110, 0b0001, 0b0100, 0b0011};


extern char throwDice(void);
# 50 "mcc_generated_files/pin_manager.c" 2

# 1 "mcc_generated_files/tmr2.h" 1
# 54 "mcc_generated_files/tmr2.h"
# 1 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdint.h" 1 3
# 54 "mcc_generated_files/tmr2.h" 2

# 1 "C:\\Program Files\\Microchip\\xc8\\v2.30\\pic\\include\\c90\\stdbool.h" 1 3
# 55 "mcc_generated_files/tmr2.h" 2
# 103 "mcc_generated_files/tmr2.h"
void TMR2_Initialize(void);
# 132 "mcc_generated_files/tmr2.h"
void TMR2_StartTimer(void);
# 164 "mcc_generated_files/tmr2.h"
void TMR2_StopTimer(void);
# 51 "mcc_generated_files/pin_manager.c" 2





void (*IOCAF3_InterruptHandler)(void);


void PIN_MANAGER_Initialize(void)
{



    LATA = 0x00;




    TRISA = 0x00;




    ANSELA = 0x00;




    WPUA = 0x00;
    OPTION_REGbits.nWPUEN = 1;







    IOCAFbits.IOCAF3 = 0;

    IOCANbits.IOCAN3 = 1;

    IOCAPbits.IOCAP3 = 1;






    INTCONbits.IOCIE = 1;

}
