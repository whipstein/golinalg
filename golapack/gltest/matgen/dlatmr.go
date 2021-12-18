package matgen

import (
	"fmt"
	"math"

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
func Dlatmr(m, n int, dist byte, iseed *[]int, sym byte, d *mat.Vector, mode int, cond, dmax float64, rsign, grade byte, dl *mat.Vector, model int, condl float64, dr *mat.Vector, moder int, condr float64, pivtng byte, ipivot *[]int, kl, ku int, sparse, anorm float64, pack byte, a *mat.Matrix, iwork *[]int) (info int, err error) {
	var badpvt, dzero, fulbnd bool
	var alpha, one, onorm, temp, zero float64
	var i, idist, igrade, iisub, ipack, ipvtng, irsign, isub, isym, j, jjsub, jsub, k, kll, kuu, mnmin, mnsub, mxsub, npvts int

	tempa := vf(1)

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	if m == 0 || n == 0 {
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
		npvts = m
	} else if pivtng == 'R' {
		ipvtng = 2
		npvts = n
	} else if pivtng == 'B' {
		ipvtng = 3
		npvts = min(n, m)
	} else if pivtng == 'F' {
		ipvtng = 3
		npvts = min(n, m)
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
	mnmin = min(m, n)
	kll = min(kl, m-1)
	kuu = min(ku, n-1)

	//     If inv(DL) is used, check to see if DL has a zero entry.
	dzero = false
	if igrade == 4 && model == 0 {
		for i = 1; i <= m; i++ {
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
	if m < 0 {
		info = -1
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if m != n && isym == 0 {
		info = -1
		err = fmt.Errorf("m != n && isym == 0: m=%v, n=%v, sym='%c'", m, n, sym)
	} else if n < 0 {
		info = -2
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if idist == -1 {
		info = -3
		err = fmt.Errorf("idist == -1: dist='%c'", dist)
	} else if isym == -1 {
		info = -5
		err = fmt.Errorf("isym == -1: sym='%c'", sym)
	} else if mode < -6 || mode > 6 {
		info = -7
		err = fmt.Errorf("mode < -6 || mode > 6: mode=%v", mode)
	} else if (mode != -6 && mode != 0 && mode != 6) && cond < one {
		info = -8
		err = fmt.Errorf("(mode != -6 && mode != 0 && mode != 6) && cond < one: mode=%v, cond=%v", mode, cond)
	} else if (mode != -6 && mode != 0 && mode != 6) && irsign == -1 {
		info = -10
		err = fmt.Errorf("(mode != -6 && mode != 0 && mode != 6) && irsign == -1: mode=%v, rsign='%c'", mode, rsign)
	} else if igrade == -1 || (igrade == 4 && m != n) || ((igrade >= 1 && igrade <= 4) && isym == 0) {
		info = -11
		err = fmt.Errorf("igrade == -1 || (igrade == 4 && m != n) || ((igrade >= 1 && igrade <= 4) && isym == 0): grade='%c', sym='%c'", grade, sym)
	} else if igrade == 4 && dzero {
		info = -12
		err = fmt.Errorf("igrade == 4 && dzero: grade='%c'", grade)
	} else if (igrade == 1 || igrade == 3 || igrade == 4 || igrade == 5) && (model < -6 || model > 6) {
		info = -13
		err = fmt.Errorf("(igrade == 1 || igrade == 3 || igrade == 4 || igrade == 5) && (model < -6 || model > 6): grade='%c', model=%v", grade, model)
	} else if (igrade == 1 || igrade == 3 || igrade == 4 || igrade == 5) && (model != -6 && model != 0 && model != 6) && condl < one {
		info = -14
		err = fmt.Errorf("(igrade == 1 || igrade == 3 || igrade == 4 || igrade == 5) && (model != -6 && model != 0 && model != 6) && condl < one: grade='%c', model=%v, condl=%v", grade, model, condl)
	} else if (igrade == 2 || igrade == 3) && (moder < -6 || moder > 6) {
		info = -16
		err = fmt.Errorf("(igrade == 2 || igrade == 3) && (moder < -6 || moder > 6): grade='%c', moder=%v", grade, moder)
	} else if (igrade == 2 || igrade == 3) && (moder != -6 && moder != 0 && moder != 6) && condr < one {
		info = -17
		err = fmt.Errorf("(igrade == 2 || igrade == 3) && (moder != -6 && moder != 0 && moder != 6) && condr < one: grade='%c', moder=%v, condr=%v", grade, moder, condr)
	} else if ipvtng == -1 || (ipvtng == 3 && m != n) || ((ipvtng == 1 || ipvtng == 2) && isym == 0) {
		info = -18
		err = fmt.Errorf("ipvtng == -1 || (ipvtng == 3 && m != n) || ((ipvtng == 1 || ipvtng == 2) && isym == 0): ipvtng=%v, m=%v, n=%v, sym='%c'", ipvtng, m, n, sym)
	} else if ipvtng != 0 && badpvt {
		info = -19
		err = fmt.Errorf("ipvtng != 0 && badpvt: ipvtng=%v, badpvt=%v", ipvtng, badpvt)
	} else if kl < 0 {
		info = -20
		err = fmt.Errorf("kl < 0: kl=%v", kl)
	} else if ku < 0 || (isym == 0 && kl != ku) {
		info = -21
		err = fmt.Errorf("ku < 0 || (isym == 0 && kl != ku): kl=%v, ku=%v, sym='%c'", kl, ku, sym)
	} else if sparse < zero || sparse > one {
		info = -22
		err = fmt.Errorf("sparse < zero || sparse > one: sparse=%v", sparse)
	} else if ipack == -1 || ((ipack == 1 || ipack == 2 || ipack == 5 || ipack == 6) && isym == 1) || (ipack == 3 && isym == 1 && (kl != 0 || m != n)) || (ipack == 4 && isym == 1 && (ku != 0 || m != n)) {
		info = -24
		err = fmt.Errorf("ipack == -1 || ((ipack == 1 || ipack == 2 || ipack == 5 || ipack == 6) && isym == 1) || (ipack == 3 && isym == 1 && (kl != 0 || m != n)) || (ipack == 4 && isym == 1 && (ku != 0 || m != n)): pack='%c', sym='%c', m=%v, n=%v, kl=%v, ku=%v", pack, sym, m, n, kl, ku)
	} else if ((ipack == 0 || ipack == 1 || ipack == 2) && a.Rows < max(1, m)) || ((ipack == 3 || ipack == 4) && a.Rows < 1) || ((ipack == 5 || ipack == 6) && a.Rows < kuu+1) || (ipack == 7 && a.Rows < kll+kuu+1) {
		info = -26
		err = fmt.Errorf("((ipack == 0 || ipack == 1 || ipack == 2) && a.Rows < max(1, m)) || ((ipack == 3 || ipack == 4) && a.Rows < 1) || ((ipack == 5 || ipack == 6) && a.Rows < kuu+1) || (ipack == 7 && a.Rows < kll+kuu+1): pack='%c', a.Rows=%v, m=%v, n=%v, kl=%v, ku=%v", pack, a.Rows, m, n, kl, ku)
	}

	if err != nil {
		gltest.Xerbla2("DLATMR", err)
		return
	}

	//     Decide if we can pivot consistently
	fulbnd = false
	if kuu == n-1 && kll == m-1 {
		fulbnd = true
	}

	//     Initialize random number generator
	for i = 1; i <= 4; i++ {
		(*iseed)[i-1] = abs((*iseed)[i-1]) % 4096
	}

	(*iseed)[3] = 2*((*iseed)[3]/2) + 1

	//     2)      Set up D, DL, and DR, if indicated.
	//
	//             Compute D according to COND and MODE
	if err = Dlatm1(mode, cond, irsign, idist, iseed, d, mnmin); err != nil {
		info = 1
		return
	}
	if mode != 0 && mode != -6 && mode != 6 {
		//        Scale by DMAX
		temp = math.Abs(d.Get(0))
		for i = 2; i <= mnmin; i++ {
			temp = math.Max(temp, math.Abs(d.Get(i-1)))
		}
		if temp == zero && dmax != zero {
			info = 2
			return
		}
		if temp != zero {
			alpha = dmax / temp
		} else {
			alpha = one
		}
		for i = 1; i <= mnmin; i++ {
			d.Set(i-1, alpha*d.Get(i-1))
		}

	}

	//     Compute DL if grading set
	if igrade == 1 || igrade == 3 || igrade == 4 || igrade == 5 {
		if err = Dlatm1(model, condl, 0, idist, iseed, dl, m); err != nil {
			info = 3
			return
		}
	}

	//     Compute DR if grading set
	if igrade == 2 || igrade == 3 {
		if err = Dlatm1(moder, condr, 0, idist, iseed, dr, n); err != nil {
			info = 4
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
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						isub, jsub, temp = Dlatm3(m, n, i, j, isub, jsub, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse)
						a.Set(isub-1, jsub-1, temp)
						a.Set(jsub-1, isub-1, temp)
					}
				}
			} else if isym == 1 {
				for j = 1; j <= n; j++ {
					for i = 1; i <= m; i++ {
						isub, jsub, temp = Dlatm3(m, n, i, j, isub, jsub, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse)
						a.Set(isub-1, jsub-1, temp)
					}
				}
			}

		} else if ipack == 1 {

			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					isub, jsub, temp = Dlatm3(m, n, i, j, isub, jsub, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse)
					mnsub = min(isub, jsub)
					mxsub = max(isub, jsub)
					a.Set(mnsub-1, mxsub-1, temp)
					if mnsub != mxsub {
						a.Set(mxsub-1, mnsub-1, zero)
					}
				}
			}

		} else if ipack == 2 {

			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					isub, jsub, temp = Dlatm3(m, n, i, j, isub, jsub, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse)
					mnsub = min(isub, jsub)
					mxsub = max(isub, jsub)
					a.Set(mxsub-1, mnsub-1, temp)
					if mnsub != mxsub {
						a.Set(mnsub-1, mxsub-1, zero)
					}
				}
			}

		} else if ipack == 3 {

			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					isub, jsub, temp = Dlatm3(m, n, i, j, isub, jsub, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse)

					//                 Compute K = location of (ISUB,JSUB) entry in packed
					//                 array
					mnsub = min(isub, jsub)
					mxsub = max(isub, jsub)
					k = mxsub*(mxsub-1)/2 + mnsub

					//                 Convert K to (IISUB,JJSUB) location
					jjsub = (k-1)/a.Rows + 1
					iisub = k - a.Rows*(jjsub-1)

					a.Set(iisub-1, jjsub-1, temp)
				}
			}

		} else if ipack == 4 {

			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					isub, jsub, temp = Dlatm3(m, n, i, j, isub, jsub, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse)

					//                 Compute K = location of (I,J) entry in packed array
					mnsub = min(isub, jsub)
					mxsub = max(isub, jsub)
					if mnsub == 1 {
						k = mxsub
					} else {
						k = n*(n+1)/2 - (n-mnsub+1)*(n-mnsub+2)/2 + mxsub - mnsub + 1
					}

					//                 Convert K to (IISUB,JJSUB) location
					jjsub = (k-1)/a.Rows + 1
					iisub = k - a.Rows*(jjsub-1)

					a.Set(iisub-1, jjsub-1, temp)
				}
			}

		} else if ipack == 5 {

			for j = 1; j <= n; j++ {
				for i = j - kuu; i <= j; i++ {
					if i < 1 {
						a.Set(j-i, i+n-1, zero)
					} else {
						isub, jsub, temp = Dlatm3(m, n, i, j, isub, jsub, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse)
						mnsub = min(isub, jsub)
						mxsub = max(isub, jsub)
						a.Set(mxsub-mnsub, mnsub-1, temp)
					}
				}
			}

		} else if ipack == 6 {

			for j = 1; j <= n; j++ {
				for i = j - kuu; i <= j; i++ {
					isub, jsub, temp = Dlatm3(m, n, i, j, isub, jsub, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse)
					mnsub = min(isub, jsub)
					mxsub = max(isub, jsub)
					a.Set(mnsub-mxsub+kuu, mxsub-1, temp)
				}
			}

		} else if ipack == 7 {

			if isym == 0 {
				for j = 1; j <= n; j++ {
					for i = j - kuu; i <= j; i++ {
						isub, jsub, temp = Dlatm3(m, n, i, j, isub, jsub, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse)
						mnsub = min(isub, jsub)
						mxsub = max(isub, jsub)
						a.Set(mnsub-mxsub+kuu, mxsub-1, temp)
						if i < 1 {
							a.Set(j-i+1+kuu-1, i+n-1, zero)
						}
						if i >= 1 && mnsub != mxsub {
							a.Set(mxsub-mnsub+1+kuu-1, mnsub-1, temp)
						}
					}
				}
			} else if isym == 1 {
				for j = 1; j <= n; j++ {
					for i = j - kuu; i <= j+kll; i++ {
						isub, jsub, temp = Dlatm3(m, n, i, j, isub, jsub, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse)
						a.Set(isub-jsub+kuu, jsub-1, temp)
					}
				}
			}

		}

	} else {
		//        Use DLATM2
		if ipack == 0 {
			if isym == 0 {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						a.Set(i-1, j-1, Dlatm2(m, n, i, j, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse))
						a.Set(j-1, i-1, a.Get(i-1, j-1))
					}
				}
			} else if isym == 1 {
				for j = 1; j <= n; j++ {
					for i = 1; i <= m; i++ {
						a.Set(i-1, j-1, Dlatm2(m, n, i, j, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse))
					}
				}
			}

		} else if ipack == 1 {

			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					a.Set(i-1, j-1, Dlatm2(m, n, i, j, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse))
					if i != j {
						a.Set(j-1, i-1, zero)
					}
				}
			}

		} else if ipack == 2 {

			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					a.Set(j-1, i-1, Dlatm2(m, n, i, j, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse))
					if i != j {
						a.Set(i-1, j-1, zero)
					}
				}
			}

		} else if ipack == 3 {

			isub = 0
			jsub = 1
			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					isub = isub + 1
					if isub > a.Rows {
						isub = 1
						jsub = jsub + 1
					}
					a.Set(isub-1, jsub-1, Dlatm2(m, n, i, j, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse))
				}
			}

		} else if ipack == 4 {

			if isym == 0 {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						//                    Compute K = location of (I,J) entry in packed array
						if i == 1 {
							k = j
						} else {
							k = n*(n+1)/2 - (n-i+1)*(n-i+2)/2 + j - i + 1
						}

						//                    Convert K to (ISUB,JSUB) location
						jsub = (k-1)/a.Rows + 1
						isub = k - a.Rows*(jsub-1)

						a.Set(isub-1, jsub-1, Dlatm2(m, n, i, j, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse))
					}
				}
			} else {
				isub = 0
				jsub = 1
				for j = 1; j <= n; j++ {
					for i = j; i <= m; i++ {
						isub = isub + 1
						if isub > a.Rows {
							isub = 1
							jsub = jsub + 1
						}
						a.Set(isub-1, jsub-1, Dlatm2(m, n, i, j, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse))
					}
				}
			}

		} else if ipack == 5 {

			for j = 1; j <= n; j++ {
				for i = j - kuu; i <= j; i++ {
					if i < 1 {
						a.Set(j-i, i+n-1, zero)
					} else {
						a.Set(j-i, i-1, Dlatm2(m, n, i, j, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse))
					}
				}
			}

		} else if ipack == 6 {

			for j = 1; j <= n; j++ {
				for i = j - kuu; i <= j; i++ {
					a.Set(i-j+kuu, j-1, Dlatm2(m, n, i, j, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse))
				}
			}

		} else if ipack == 7 {

			if isym == 0 {
				for j = 1; j <= n; j++ {
					for i = j - kuu; i <= j; i++ {
						a.Set(i-j+kuu, j-1, Dlatm2(m, n, i, j, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse))
						if i < 1 {
							a.Set(j-i+1+kuu-1, i+n-1, zero)
						}
						if i >= 1 && i != j {
							a.Set(j-i+1+kuu-1, i-1, a.Get(i-j+kuu, j-1))
						}
					}
				}
			} else if isym == 1 {
				for j = 1; j <= n; j++ {
					for i = j - kuu; i <= j+kll; i++ {
						a.Set(i-j+kuu, j-1, Dlatm2(m, n, i, j, kl, ku, idist, iseed, d, igrade, dl, dr, ipvtng, iwork, sparse))
					}
				}
			}

		}

	}

	//     5)      Scaling the norm
	if ipack == 0 {
		onorm = golapack.Dlange('M', m, n, a, tempa)
	} else if ipack == 1 {
		onorm = golapack.Dlansy('M', Upper, n, a, tempa)
	} else if ipack == 2 {
		onorm = golapack.Dlansy('M', Lower, n, a, tempa)
	} else if ipack == 3 {
		onorm = golapack.Dlansp('M', Upper, n, a.OffIdx(0).Vector(), tempa)
	} else if ipack == 4 {
		onorm = golapack.Dlansp('M', Lower, n, a.OffIdx(0).Vector(), tempa)
	} else if ipack == 5 {
		onorm = golapack.Dlansb('M', Lower, n, kll, a, tempa)
	} else if ipack == 6 {
		onorm = golapack.Dlansb('M', Upper, n, kuu, a, tempa)
	} else if ipack == 7 {
		onorm = golapack.Dlangb('M', n, kll, kuu, a, tempa)
	}

	if anorm >= zero {

		if anorm > zero && onorm == zero {
			//           Desired scaling impossible
			info = 5
			return

		} else if (anorm > one && onorm < one) || (anorm < one && onorm > one) {
			//           Scale carefully to avoid over / underflow
			if ipack <= 2 {
				for j = 1; j <= n; j++ {
					a.Off(0, j-1).Vector().Scal(m, one/onorm, 1)
					a.Off(0, j-1).Vector().Scal(m, anorm, 1)
				}

			} else if ipack == 3 || ipack == 4 {

				a.OffIdx(0).Vector().Scal(n*(n+1)/2, one/onorm, 1)
				a.OffIdx(0).Vector().Scal(n*(n+1)/2, anorm, 1)
			} else if ipack >= 5 {

				for j = 1; j <= n; j++ {
					a.Off(0, j-1).Vector().Scal(kll+kuu+1, one/onorm, 1)
					a.Off(0, j-1).Vector().Scal(kll+kuu+1, anorm, 1)
				}

			}

		} else {
			//           Scale straightforwardly
			if ipack <= 2 {
				for j = 1; j <= n; j++ {
					a.Off(0, j-1).Vector().Scal(m, anorm/onorm, 1)
				}

			} else if ipack == 3 || ipack == 4 {

				a.OffIdx(0).Vector().Scal(n*(n+1)/2, anorm/onorm, 1)

			} else if ipack >= 5 {

				for j = 1; j <= n; j++ {
					a.Off(0, j-1).Vector().Scal(kll+kuu+1, anorm/onorm, 1)
				}
			}

		}

	}

	return
}
