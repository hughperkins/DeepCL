

// concept:
// - load same input plane from each image
// - hold filter plane for this input plane, for all filters
// - reduce afterwards
// local memory for one plane from each filter of 64c7 = 64 * 7 * 7 * 4 = 12.5KB
// local memory for one single input plane = 19 * 19 * 4 = 1.4KB
// => seems ok?

