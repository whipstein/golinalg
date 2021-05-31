package matgen

import "github.com/whipstein/golinalg/mat"

// Dlatm2 returns the (I,J) entry of a random matrix of dimension
//    (M, N) described by the other parameters. It is called by the
//    DLATMR routine in order to build random test matrices. No error
//    checking on parameters is done, because this routine is called in
//    a tight loop by DLATMR which has already checked the parameters.
//
//    Use of DLATM2 differs from SLATM3 in the order in which the random
//    number generator is called to fill in random matrix entries.
//    With DLATM2, the generator is called to fill in the pivoted matrix
//    columnwise. With DLATM3, the generator is called to fill in the
//    matrix columnwise, after which it is pivoted. Thus, DLATM3 can
//    be used to construct random matrices which differ only in their
//    order of rows and/or columns. DLATM2 is used to construct band
//    matrices while avoiding calling the random number generator for
//    entries outside the band (and therefore generating random numbers
//
//    The matrix whose (I,J) entry is returned is constructed as
//    follows (this routine only computes one entry):
//
//      If I is outside (1..M) or J is outside (1..N), return zero
//         (this is convenient for generating matrices in band format).
//
//      Generate a matrix A with random entries of distribution IDIST.
//
//      Set the diagonal to D.
//
//      Grade the matrix, if desired, from the left (by DL) and/or
//         from the right (by DR or DL) as specified by IGRADE.
//
//      Permute, if desired, the rows and/or columns as specified by
//         IPVTNG and IWORK.
//
//      Band the matrix to have lower bandwidth KL and upper
//         bandwidth KU.
//
//      Set random entries to zero as specified by SPARSE.
func Dlatm2(m, n, i, j, kl, ku, idist *int, iseed *[]int, d *mat.Vector, igrade *int, dl, dr *mat.Vector, ipvtng *int, iwork *[]int, sparse *float64) (dlatm2Return float64) {
	var temp, zero float64
	var isub, jsub int

	zero = 0.0

	//     Check for I and J in range
	if (*i) < 1 || (*i) > (*m) || (*j) < 1 || (*j) > (*n) {
		dlatm2Return = zero
		return
	}

	//     Check for banding
	if (*j) > (*i)+(*ku) || (*j) < (*i)-(*kl) {
		dlatm2Return = zero
		return
	}

	//     Check for sparsity
	if (*sparse) > zero {
		if Dlaran(iseed) < (*sparse) {
			dlatm2Return = zero
			return
		}
	}

	//     Compute subscripts depending on IPVTNG
	if (*ipvtng) == 0 {
		isub = (*i)
		jsub = (*j)
	} else if (*ipvtng) == 1 {
		isub = (*iwork)[(*i)-1]
		jsub = (*j)
	} else if (*ipvtng) == 2 {
		isub = (*i)
		jsub = (*iwork)[(*j)-1]
	} else if (*ipvtng) == 3 {
		isub = (*iwork)[(*i)-1]
		jsub = (*iwork)[(*j)-1]
	}

	//     Compute entry and grade it according to IGRADE
	if isub == jsub {
		temp = d.Get(isub - 1)
	} else {
		temp = Dlarnd(idist, iseed)
	}
	if (*igrade) == 1 {
		temp = temp * dl.Get(isub-1)
	} else if (*igrade) == 2 {
		temp = temp * dr.Get(jsub-1)
	} else if (*igrade) == 3 {
		temp = temp * dl.Get(isub-1) * dr.Get(jsub-1)
	} else if (*igrade) == 4 && isub != jsub {
		temp = temp * dl.Get(isub-1) / dl.Get(jsub-1)
	} else if (*igrade) == 5 {
		temp = temp * dl.Get(isub-1) * dl.Get(jsub-1)
	}
	dlatm2Return = temp
	return
}
