package matgen

import "github.com/whipstein/golinalg/mat"

// Zlatm2 returns the (I,J) entry of a random matrix of dimension
//    (M, N) described by the other parameters. It is called by the
//    ZLATMR routine in order to build random test matrices. No error
//    checking on parameters is done, because this routine is called in
//    a tight loop by ZLATMR which has already checked the parameters.
//
//    Use of ZLATM2 differs from CLATM3 in the order in which the random
//    number generator is called to fill in random matrix entries.
//    With ZLATM2, the generator is called to fill in the pivoted matrix
//    columnwise. With ZLATM3, the generator is called to fill in the
//    matrix columnwise, after which it is pivoted. Thus, ZLATM3 can
//    be used to construct random matrices which differ only in their
//    order of rows and/or columns. ZLATM2 is used to construct band
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
func Zlatm2(m, n, i, j, kl, ku, idist *int, iseed *[]int, d *mat.CVector, igrade *int, dl, dr *mat.CVector, ipvtng *int, iwork *[]int, sparse *float64) (zlatm2Return complex128) {
	var ctemp, czero complex128
	var zero float64
	var isub, jsub int

	czero = (0.0 + 0.0*1i)
	zero = 0.0

	//     Check for I and J in range
	if (*i) < 1 || (*i) > (*m) || (*j) < 1 || (*j) > (*n) {
		zlatm2Return = czero
		return
	}

	//     Check for banding
	if (*j) > (*i)+(*ku) || (*j) < (*i)-(*kl) {
		zlatm2Return = czero
		return
	}

	//     Check for sparsity
	if (*sparse) > zero {
		if Dlaran(iseed) < (*sparse) {
			zlatm2Return = czero
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
		ctemp = d.Get(isub - 1)
	} else {
		ctemp = Zlarnd(idist, iseed)
	}
	if (*igrade) == 1 {
		ctemp = ctemp * dl.Get(isub-1)
	} else if (*igrade) == 2 {
		ctemp = ctemp * dr.Get(jsub-1)
	} else if (*igrade) == 3 {
		ctemp = ctemp * dl.Get(isub-1) * dr.Get(jsub-1)
	} else if (*igrade) == 4 && isub != jsub {
		ctemp = ctemp * dl.Get(isub-1) / dl.Get(jsub-1)
	} else if (*igrade) == 5 {
		ctemp = ctemp * dl.Get(isub-1) * dl.GetConj(jsub-1)
	} else if (*igrade) == 6 {
		ctemp = ctemp * dl.Get(isub-1) * dl.Get(jsub-1)
	}
	zlatm2Return = ctemp
	return
}
