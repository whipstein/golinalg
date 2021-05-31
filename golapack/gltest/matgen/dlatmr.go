package matgen

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlatmr generates random matrices of various types for testing
//    LAPACK programs.
//
//    DLATMR operates by applying the following sequence of
//    operations:
//
//      Generate a matrix A with random entries of distribution DIST
//         which is symmetric if SYM='S', and nonsymmetric
//         if SYM='N'.
//
//      Set the diagonal to D, where D may be input or
//         computed according to MODE, COND, DMAX and RSIGN
//         as described below.
//
//      Grade the matrix, if desired, from the left and/or right
//         as specified by GRADE. The inputs DL, MODEL, CONDL, DR,
//         MODER and CONDR also determine the grading as described
//         below.
//
//      Permute, if desired, the rows and/or columns as specified by
//         PIVTNG and IPIVOT.
//
//      Set random entries to zero, if desired, to get a random sparse
//         matrix as specified by SPARSE.
//
//      Make A a band matrix, if desired, by zeroing out the matrix
//         outside a band of lower bandwidth KL and upper bandwidth KU.
//
//      Scale A, if desired, to have maximum entry ANORM.
//
//      Pack the matrix if desired. Options specified by PACK are:
//         no packing
//         zero out upper half (if symmetric)
//         zero out lower half (if symmetric)
//         store the upper half columnwise (if symmetric or
//             square upper triangular)
//         store the lower half columnwise (if symmetric or
//             square lower triangular)
//             same as upper half rowwise if symmetric
//         store the lower triangle in banded format (if symmetric)
//         store the upper triangle in banded format (if symmetric)
//         store the entire matrix in banded format
//
//    Note: If two calls to DLATMR differ only in the PACK parameter,
//          they will generate mathematically equivalent matrices.
//
//          If two calls to DLATMR both have full bandwidth (KL = M-1
//          and KU = N-1), and differ only in the PIVTNG and PACK
//          parameters, then the matrices generated will differ only
//          in the order of the rows and/or columns, and otherwise
//          contain the same data. This consistency cannot be and
//          is not maintained with less than full bandwidth.
func Dlatmr(m, n *int, dist byte, iseed *[]int, sym byte, d *mat.Vector, mode *int, cond, dmax *float64, rsign, grade byte, dl *mat.Vector, model *int, condl *float64, dr *mat.Vector, moder *int, condr *float64, pivtng byte, ipivot *[]int, kl, ku *int, sparse, anorm *float64, pack byte, a *mat.Matrix, lda *int, iwork *[]int, info *int) {
	var badpvt, dzero, fulbnd bool
	var alpha, one, onorm, temp, zero float64
	var i, idist, igrade, iisub, ipack, ipvtng, irsign, isub, isym, j, jjsub, jsub, k, kll, kuu, mnmin, mnsub, mxsub, npvts int

	tempa := vf(1)

	zero = 0.0
	one = 1.0

	//     1)      Decode and Test the input parameters.
	//             Initialize flags & seed.
	(*info) = 0

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	//     Decode DIST
	if dist == 'U' {
		idist = 1
	} else if dist == 'S' {
		idist = 2
	} else if dist == 'N' {
		idist = 3
	} else {
		idist = -1
	}

	//     Decode SYM
	if sym == 'S' {
		isym = 0
	} else if sym == 'N' {
		isym = 1
	} else if sym == 'H' {
		isym = 0
	} else {
		isym = -1
	}

	//     Decode RSIGN
	if rsign == 'F' {
		irsign = 0
	} else if rsign == 'T' {
		irsign = 1
	} else {
		irsign = -1
	}

	//     Decode PIVTNG
	if pivtng == 'N' {
		ipvtng = 0
	} else if pivtng == ' ' {
		ipvtng = 0
	} else if pivtng == 'L' {
		ipvtng = 1
		npvts = (*m)
	} else if pivtng == 'R' {
		ipvtng = 2
		npvts = (*n)
	} else if pivtng == 'B' {
		ipvtng = 3
		npvts = minint(*n, *m)
	} else if pivtng == 'F' {
		ipvtng = 3
		npvts = minint(*n, *m)
	} else {
		ipvtng = -1
	}

	//     Decode GRADE
	if grade == 'N' {
		igrade = 0
	} else if grade == 'L' {
		igrade = 1
	} else if grade == 'R' {
		igrade = 2
	} else if grade == 'B' {
		igrade = 3
	} else if grade == 'E' {
		igrade = 4
	} else if grade == 'H' || grade == 'S' {
		igrade = 5
	} else {
		igrade = -1
	}

	//     Decode PACK
	if pack == 'N' {
		ipack = 0
	} else if pack == 'U' {
		ipack = 1
	} else if pack == 'L' {
		ipack = 2
	} else if pack == 'C' {
		ipack = 3
	} else if pack == 'R' {
		ipack = 4
	} else if pack == 'B' {
		ipack = 5
	} else if pack == 'Q' {
		ipack = 6
	} else if pack == 'Z' {
		ipack = 7
	} else {
		ipack = -1
	}

	//     Set certain internal parameters
	mnmin = minint(*m, *n)
	kll = minint(*kl, (*m)-1)
	kuu = minint(*ku, (*n)-1)

	//     If inv(DL) is used, check to see if DL has a zero entry.
	dzero = false
	if igrade == 4 && (*model) == 0 {
		for i = 1; i <= (*m); i++ {
			if dl.Get(i-1) == zero {
				dzero = true
			}
		}
	}

	//     Check values in IPIVOT
	badpvt = false
	if ipvtng > 0 {
		for j = 1; j <= npvts; j++ {
			if (*ipivot)[j-1] <= 0 || (*ipivot)[j-1] > npvts {
				badpvt = true
			}
		}
	}

	//     Set INFO if an error
	if (*m) < 0 {
		(*info) = -1
	} else if (*m) != (*n) && isym == 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if idist == -1 {
		(*info) = -3
	} else if isym == -1 {
		(*info) = -5
	} else if (*mode) < -6 || (*mode) > 6 {
		(*info) = -7
	} else if ((*mode) != -6 && (*mode) != 0 && (*mode) != 6) && (*cond) < one {
		(*info) = -8
	} else if ((*mode) != -6 && (*mode) != 0 && (*mode) != 6) && irsign == -1 {
		(*info) = -10
	} else if igrade == -1 || (igrade == 4 && (*m) != (*n)) || ((igrade >= 1 && igrade <= 4) && isym == 0) {
		(*info) = -11
	} else if igrade == 4 && dzero {
		(*info) = -12
	} else if (igrade == 1 || igrade == 3 || igrade == 4 || igrade == 5) && ((*model) < -6 || (*model) > 6) {
		(*info) = -13
	} else if (igrade == 1 || igrade == 3 || igrade == 4 || igrade == 5) && ((*model) != -6 && (*model) != 0 && (*model) != 6) && (*condl) < one {
		(*info) = -14
	} else if (igrade == 2 || igrade == 3) && ((*moder) < -6 || (*moder) > 6) {
		(*info) = -16
	} else if (igrade == 2 || igrade == 3) && ((*moder) != -6 && (*moder) != 0 && (*moder) != 6) && (*condr) < one {
		(*info) = -17
	} else if ipvtng == -1 || (ipvtng == 3 && (*m) != (*n)) || ((ipvtng == 1 || ipvtng == 2) && isym == 0) {
		(*info) = -18
	} else if ipvtng != 0 && badpvt {
		(*info) = -19
	} else if (*kl) < 0 {
		(*info) = -20
	} else if (*ku) < 0 || (isym == 0 && (*kl) != (*ku)) {
		(*info) = -21
	} else if (*sparse) < zero || (*sparse) > one {
		(*info) = -22
	} else if ipack == -1 || ((ipack == 1 || ipack == 2 || ipack == 5 || ipack == 6) && isym == 1) || (ipack == 3 && isym == 1 && ((*kl) != 0 || (*m) != (*n))) || (ipack == 4 && isym == 1 && ((*ku) != 0 || (*m) != (*n))) {
		(*info) = -24
	} else if ((ipack == 0 || ipack == 1 || ipack == 2) && (*lda) < maxint(1, *m)) || ((ipack == 3 || ipack == 4) && (*lda) < 1) || ((ipack == 5 || ipack == 6) && (*lda) < kuu+1) || (ipack == 7 && (*lda) < kll+kuu+1) {
		(*info) = -26
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DLATMR"), -(*info))
		return
	}

	//     Decide if we can pivot consistently
	fulbnd = false
	if kuu == (*n)-1 && kll == (*m)-1 {
		fulbnd = true
	}

	//     Initialize random number generator
	for i = 1; i <= 4; i++ {
		(*iseed)[i-1] = absint((*iseed)[i-1]) % 4096
	}

	(*iseed)[3] = 2*((*iseed)[3]/2) + 1

	//     2)      Set up D, DL, and DR, if indicated.
	//
	//             Compute D according to COND and MODE
	Dlatm1(mode, cond, &irsign, &idist, iseed, d, &mnmin, info)
	if (*info) != 0 {
		(*info) = 1
		return
	}
	if (*mode) != 0 && (*mode) != -6 && (*mode) != 6 {
		//        Scale by DMAX
		temp = math.Abs(d.Get(0))
		for i = 2; i <= mnmin; i++ {
			temp = maxf64(temp, math.Abs(d.Get(i-1)))
		}
		if temp == zero && (*dmax) != zero {
			(*info) = 2
			return
		}
		if temp != zero {
			alpha = (*dmax) / temp
		} else {
			alpha = one
		}
		for i = 1; i <= mnmin; i++ {
			d.Set(i-1, alpha*d.Get(i-1))
		}

	}

	//     Compute DL if grading set
	if igrade == 1 || igrade == 3 || igrade == 4 || igrade == 5 {
		Dlatm1(model, condl, func() *int { y := 0; return &y }(), &idist, iseed, dl, m, info)
		if (*info) != 0 {
			(*info) = 3
			return
		}
	}

	//     Compute DR if grading set
	if igrade == 2 || igrade == 3 {
		Dlatm1(moder, condr, func() *int { y := 0; return &y }(), &idist, iseed, dr, n, info)
		if (*info) != 0 {
			(*info) = 4
			return
		}
	}

	//     3)     Generate IWORK if pivoting
	if ipvtng > 0 {
		for i = 1; i <= npvts; i++ {
			(*iwork)[i-1] = i
		}
		if fulbnd {
			for i = 1; i <= npvts; i++ {
				k = (*ipivot)[i-1]
				j = (*iwork)[i-1]
				(*iwork)[i-1] = (*iwork)[k-1]
				(*iwork)[k-1] = j
			}
		} else {
			for i = npvts; i >= 1; i-- {
				k = (*ipivot)[i-1]
				j = (*iwork)[i-1]
				(*iwork)[i-1] = (*iwork)[k-1]
				(*iwork)[k-1] = j
			}
		}
	}

	//     4)      Generate matrices for each kind of PACKing
	//             Always sweep matrix columnwise (if symmetric, upper
	//             half only) so that matrix generated does not depend
	//             on PACK
	if fulbnd {
		//        Use DLATM3 so matrices generated with differing PIVOTing only
		//        differ only in the order of their rows and/or columns.
		if ipack == 0 {
			if isym == 0 {
				for j = 1; j <= (*n); j++ {
					for i = 1; i <= j; i++ {
						temp = Dlatm3(m, n, &i, &j, &isub, &jsub, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse)
						a.Set(isub-1, jsub-1, temp)
						a.Set(jsub-1, isub-1, temp)
					}
				}
			} else if isym == 1 {
				for j = 1; j <= (*n); j++ {
					for i = 1; i <= (*m); i++ {
						temp = Dlatm3(m, n, &i, &j, &isub, &jsub, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse)
						a.Set(isub-1, jsub-1, temp)
					}
				}
			}

		} else if ipack == 1 {

			for j = 1; j <= (*n); j++ {
				for i = 1; i <= j; i++ {
					temp = Dlatm3(m, n, &i, &j, &isub, &jsub, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse)
					mnsub = minint(isub, jsub)
					mxsub = maxint(isub, jsub)
					a.Set(mnsub-1, mxsub-1, temp)
					if mnsub != mxsub {
						a.Set(mxsub-1, mnsub-1, zero)
					}
				}
			}

		} else if ipack == 2 {

			for j = 1; j <= (*n); j++ {
				for i = 1; i <= j; i++ {
					temp = Dlatm3(m, n, &i, &j, &isub, &jsub, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse)
					mnsub = minint(isub, jsub)
					mxsub = maxint(isub, jsub)
					a.Set(mxsub-1, mnsub-1, temp)
					if mnsub != mxsub {
						a.Set(mnsub-1, mxsub-1, zero)
					}
				}
			}

		} else if ipack == 3 {

			for j = 1; j <= (*n); j++ {
				for i = 1; i <= j; i++ {
					temp = Dlatm3(m, n, &i, &j, &isub, &jsub, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse)

					//                 Compute K = location of (ISUB,JSUB) entry in packed
					//                 array
					mnsub = minint(isub, jsub)
					mxsub = maxint(isub, jsub)
					k = mxsub*(mxsub-1)/2 + mnsub

					//                 Convert K to (IISUB,JJSUB) location
					jjsub = (k-1)/(*lda) + 1
					iisub = k - (*lda)*(jjsub-1)

					a.Set(iisub-1, jjsub-1, temp)
				}
			}

		} else if ipack == 4 {

			for j = 1; j <= (*n); j++ {
				for i = 1; i <= j; i++ {
					temp = Dlatm3(m, n, &i, &j, &isub, &jsub, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse)

					//                 Compute K = location of (I,J) entry in packed array
					mnsub = minint(isub, jsub)
					mxsub = maxint(isub, jsub)
					if mnsub == 1 {
						k = mxsub
					} else {
						k = (*n)*((*n)+1)/2 - ((*n)-mnsub+1)*((*n)-mnsub+2)/2 + mxsub - mnsub + 1
					}

					//                 Convert K to (IISUB,JJSUB) location
					jjsub = (k-1)/(*lda) + 1
					iisub = k - (*lda)*(jjsub-1)

					a.Set(iisub-1, jjsub-1, temp)
				}
			}

		} else if ipack == 5 {

			for j = 1; j <= (*n); j++ {
				for i = j - kuu; i <= j; i++ {
					if i < 1 {
						a.Set(j-i+1-1, i+(*n)-1, zero)
					} else {
						temp = Dlatm3(m, n, &i, &j, &isub, &jsub, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse)
						mnsub = minint(isub, jsub)
						mxsub = maxint(isub, jsub)
						a.Set(mxsub-mnsub+1-1, mnsub-1, temp)
					}
				}
			}

		} else if ipack == 6 {

			for j = 1; j <= (*n); j++ {
				for i = j - kuu; i <= j; i++ {
					temp = Dlatm3(m, n, &i, &j, &isub, &jsub, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse)
					mnsub = minint(isub, jsub)
					mxsub = maxint(isub, jsub)
					a.Set(mnsub-mxsub+kuu+1-1, mxsub-1, temp)
				}
			}

		} else if ipack == 7 {

			if isym == 0 {
				for j = 1; j <= (*n); j++ {
					for i = j - kuu; i <= j; i++ {
						temp = Dlatm3(m, n, &i, &j, &isub, &jsub, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse)
						mnsub = minint(isub, jsub)
						mxsub = maxint(isub, jsub)
						a.Set(mnsub-mxsub+kuu+1-1, mxsub-1, temp)
						if i < 1 {
							a.Set(j-i+1+kuu-1, i+(*n)-1, zero)
						}
						if i >= 1 && mnsub != mxsub {
							a.Set(mxsub-mnsub+1+kuu-1, mnsub-1, temp)
						}
					}
				}
			} else if isym == 1 {
				for j = 1; j <= (*n); j++ {
					for i = j - kuu; i <= j+kll; i++ {
						temp = Dlatm3(m, n, &i, &j, &isub, &jsub, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse)
						a.Set(isub-jsub+kuu+1-1, jsub-1, temp)
					}
				}
			}

		}

	} else {
		//        Use DLATM2
		if ipack == 0 {
			if isym == 0 {
				for j = 1; j <= (*n); j++ {
					for i = 1; i <= j; i++ {
						a.Set(i-1, j-1, Dlatm2(m, n, &i, &j, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse))
						a.Set(j-1, i-1, a.Get(i-1, j-1))
					}
				}
			} else if isym == 1 {
				for j = 1; j <= (*n); j++ {
					for i = 1; i <= (*m); i++ {
						a.Set(i-1, j-1, Dlatm2(m, n, &i, &j, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse))
					}
				}
			}

		} else if ipack == 1 {

			for j = 1; j <= (*n); j++ {
				for i = 1; i <= j; i++ {
					a.Set(i-1, j-1, Dlatm2(m, n, &i, &j, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse))
					if i != j {
						a.Set(j-1, i-1, zero)
					}
				}
			}

		} else if ipack == 2 {

			for j = 1; j <= (*n); j++ {
				for i = 1; i <= j; i++ {
					a.Set(j-1, i-1, Dlatm2(m, n, &i, &j, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse))
					if i != j {
						a.Set(i-1, j-1, zero)
					}
				}
			}

		} else if ipack == 3 {

			isub = 0
			jsub = 1
			for j = 1; j <= (*n); j++ {
				for i = 1; i <= j; i++ {
					isub = isub + 1
					if isub > (*lda) {
						isub = 1
						jsub = jsub + 1
					}
					a.Set(isub-1, jsub-1, Dlatm2(m, n, &i, &j, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse))
				}
			}

		} else if ipack == 4 {

			if isym == 0 {
				for j = 1; j <= (*n); j++ {
					for i = 1; i <= j; i++ {
						//                    Compute K = location of (I,J) entry in packed array
						if i == 1 {
							k = j
						} else {
							k = (*n)*((*n)+1)/2 - ((*n)-i+1)*((*n)-i+2)/2 + j - i + 1
						}

						//                    Convert K to (ISUB,JSUB) location
						jsub = (k-1)/(*lda) + 1
						isub = k - (*lda)*(jsub-1)

						a.Set(isub-1, jsub-1, Dlatm2(m, n, &i, &j, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse))
					}
				}
			} else {
				isub = 0
				jsub = 1
				for j = 1; j <= (*n); j++ {
					for i = j; i <= (*m); i++ {
						isub = isub + 1
						if isub > (*lda) {
							isub = 1
							jsub = jsub + 1
						}
						a.Set(isub-1, jsub-1, Dlatm2(m, n, &i, &j, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse))
					}
				}
			}

		} else if ipack == 5 {

			for j = 1; j <= (*n); j++ {
				for i = j - kuu; i <= j; i++ {
					if i < 1 {
						a.Set(j-i+1-1, i+(*n)-1, zero)
					} else {
						a.Set(j-i+1-1, i-1, Dlatm2(m, n, &i, &j, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse))
					}
				}
			}

		} else if ipack == 6 {

			for j = 1; j <= (*n); j++ {
				for i = j - kuu; i <= j; i++ {
					a.Set(i-j+kuu+1-1, j-1, Dlatm2(m, n, &i, &j, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse))
				}
			}

		} else if ipack == 7 {

			if isym == 0 {
				for j = 1; j <= (*n); j++ {
					for i = j - kuu; i <= j; i++ {
						a.Set(i-j+kuu+1-1, j-1, Dlatm2(m, n, &i, &j, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse))
						if i < 1 {
							a.Set(j-i+1+kuu-1, i+(*n)-1, zero)
						}
						if i >= 1 && i != j {
							a.Set(j-i+1+kuu-1, i-1, a.Get(i-j+kuu+1-1, j-1))
						}
					}
				}
			} else if isym == 1 {
				for j = 1; j <= (*n); j++ {
					for i = j - kuu; i <= j+kll; i++ {
						a.Set(i-j+kuu+1-1, j-1, Dlatm2(m, n, &i, &j, kl, ku, &idist, iseed, d, &igrade, dl, dr, &ipvtng, iwork, sparse))
					}
				}
			}

		}

	}

	//     5)      Scaling the norm
	if ipack == 0 {
		onorm = golapack.Dlange('M', m, n, a, lda, tempa)
	} else if ipack == 1 {
		onorm = golapack.Dlansy('M', 'U', n, a, lda, tempa)
	} else if ipack == 2 {
		onorm = golapack.Dlansy('M', 'L', n, a, lda, tempa)
	} else if ipack == 3 {
		onorm = golapack.Dlansp('M', 'U', n, a.VectorIdx(0), tempa)
	} else if ipack == 4 {
		onorm = golapack.Dlansp('M', 'L', n, a.VectorIdx(0), tempa)
	} else if ipack == 5 {
		onorm = golapack.Dlansb('M', 'L', n, &kll, a, lda, tempa)
	} else if ipack == 6 {
		onorm = golapack.Dlansb('M', 'U', n, &kuu, a, lda, tempa)
	} else if ipack == 7 {
		onorm = golapack.Dlangb('M', n, &kll, &kuu, a, lda, tempa)
	}

	if (*anorm) >= zero {

		if (*anorm) > zero && onorm == zero {
			//           Desired scaling impossible
			(*info) = 5
			return

		} else if ((*anorm) > one && onorm < one) || ((*anorm) < one && onorm > one) {
			//           Scale carefully to avoid over / underflow
			if ipack <= 2 {
				for j = 1; j <= (*n); j++ {
					goblas.Dscal(m, toPtrf64(one/onorm), a.Vector(0, j-1), toPtr(1))
					goblas.Dscal(m, anorm, a.Vector(0, j-1), toPtr(1))
				}

			} else if ipack == 3 || ipack == 4 {

				goblas.Dscal(toPtr((*n)*((*n)+1)/2), toPtrf64(one/onorm), a.VectorIdx(0), toPtr(1))
				goblas.Dscal(toPtr((*n)*((*n)+1)/2), anorm, a.VectorIdx(0), toPtr(1))
			} else if ipack >= 5 {

				for j = 1; j <= (*n); j++ {
					goblas.Dscal(toPtr(kll+kuu+1), toPtrf64(one/onorm), a.Vector(0, j-1), toPtr(1))
					goblas.Dscal(toPtr(kll+kuu+1), anorm, a.Vector(0, j-1), toPtr(1))
				}

			}

		} else {
			//           Scale straightforwardly
			if ipack <= 2 {
				for j = 1; j <= (*n); j++ {
					goblas.Dscal(m, toPtrf64((*anorm)/onorm), a.Vector(0, j-1), toPtr(1))
				}

			} else if ipack == 3 || ipack == 4 {

				goblas.Dscal(toPtr((*n)*((*n)+1)/2), toPtrf64((*anorm)/onorm), a.VectorIdx(0), toPtr(1))

			} else if ipack >= 5 {

				for j = 1; j <= (*n); j++ {
					goblas.Dscal(toPtr(kll+kuu+1), toPtrf64((*anorm)/onorm), a.Vector(0, j-1), toPtr(1))
				}
			}

		}

	}
}
