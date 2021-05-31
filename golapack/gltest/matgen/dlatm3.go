package matgen

import "github.com/whipstein/golinalg/mat"

// Dlatm3 returns the (ISUB,JSUB) entry of a random matrix of
//    dimension (M, N) described by the other parameters. (ISUB,JSUB)
//    is the final position of the (I,J) entry after pivoting
//    according to IPVTNG and IWORK. DLATM3 is called by the
//    DLATMR routine in order to build random test matrices. No error
//    checking on parameters is done, because this routine is called in
//    a tight loop by DLATMR which has already checked the parameters.
//
//    Use of DLATM3 differs from SLATM2 in the order in which the random
//    number generator is called to fill in random matrix entries.
//    With DLATM2, the generator is called to fill in the pivoted matrix
//    columnwise. With DLATM3, the generator is called to fill in the
//    matrix columnwise, after which it is pivoted. Thus, DLATM3 can
//    be used to construct random matrices which differ only in their
//    order of rows and/or columns. DLATM2 is used to construct band
//    matrices while avoiding calling the random number generator for
//    entries outside the band (and therefore generating random numbers
//    in different orders for different pivot orders).
//
//    The matrix whose (ISUB,JSUB) entry is returned is constructed as
//    follows (this routine only computes one entry):
//
//      If ISUB is outside (1..M) or JSUB is outside (1..N), return zero
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
func Dlatm3(m, n, i, j, isub, jsub, kl, ku, idist *int, iseed *[]int, d *mat.Vector, igrade *int, dl, dr *mat.Vector, ipvtng *int, iwork *[]int, sparse *float64) (dlatm3Return float64) {
	var temp, zero float64

	zero = 0.0

	//     Check for I and J in range
	if (*i) < 1 || (*i) > (*m) || (*j) < 1 || (*j) > (*n) {
		(*isub) = (*i)
		(*jsub) = (*j)
		dlatm3Return = zero
		return
	}

	//     Compute subscripts depending on IPVTNG
	if (*ipvtng) == 0 {
		(*isub) = (*i)
		(*jsub) = (*j)
	} else if (*ipvtng) == 1 {
		(*isub) = (*iwork)[(*i)-1]
		(*jsub) = (*j)
	} else if (*ipvtng) == 2 {
		(*isub) = (*i)
		(*jsub) = (*iwork)[(*j)-1]
	} else if (*ipvtng) == 3 {
		(*isub) = (*iwork)[(*i)-1]
		(*jsub) = (*iwork)[(*j)-1]
	}

	//     Check for banding
	if (*jsub) > (*isub)+(*ku) || (*jsub) < (*isub)-(*kl) {
		dlatm3Return = zero
		return
	}

	//     Check for sparsity
	if (*sparse) > zero {
		if Dlaran(iseed) < (*sparse) {
			dlatm3Return = zero
			return
		}
	}

	//     Compute entry and grade it according to IGRADE
	if (*i) == (*j) {
		temp = d.Get((*i) - 1)
	} else {
		temp = Dlarnd(idist, iseed)
	}
	if (*igrade) == 1 {
		temp = temp * dl.Get((*i)-1)
	} else if (*igrade) == 2 {
		temp = temp * dr.Get((*j)-1)
	} else if (*igrade) == 3 {
		temp = temp * dl.Get((*i)-1) * dr.Get((*j)-1)
	} else if (*igrade) == 4 && (*i) != (*j) {
		temp = temp * dl.Get((*i)-1) / dl.Get((*j)-1)
	} else if (*igrade) == 5 {
		temp = temp * dl.Get((*i)-1) * dl.Get((*j)-1)
	}
	dlatm3Return = temp
	return
}
