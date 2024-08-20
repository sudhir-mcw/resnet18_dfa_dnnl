#ifndef DEVICE_H
#define DEVICE_H
//The device here is a grid of 64*64
//Each PE has a size of 256 BYTES and 
//Float data that can be stored in a PE is 64  (64 * 4 = 256)

#define PE_ROWS       64
#define PE_COLUMNS    64
// Size is Equal to 1 MB 
#define SIZE_PER_PE 250000
// Size is Equal to 5 MB
// #define SIZE_PER_PE 1250000

#endif // DEVICE_H