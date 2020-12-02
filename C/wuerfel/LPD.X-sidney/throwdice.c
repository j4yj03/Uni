#include "header.h"
#include "mcc_generated_files/tmr.h"


void throwDice()
{
  tmr0val = TMR0_ReadTimer()

  tmr0val ^= tmr0val << 5;
  tmr0val ^= tmr0val >> 7;
  tmr0val ^= tmr0val << 3;

  dice.next = (tmr0val % dice.current) % dice.current;
}
