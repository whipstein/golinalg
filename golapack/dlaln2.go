package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlaln2 solves a system of the form  (ca A - w D ) X = s B
// or (ca A**T - w D) X = s B   with possible scaling ("s") and
// perturbation of A.  (A**T means A-transpose.)
//
// A is an NA x NA real matrix, ca is a real scalar, D is an NA x NA
// real diagonal matrix, w is a real or complex value, and X and B are
// NA x 1 matrices -- real if w is real, complex if w is complex.  NA
// may be 1 or 2.
//
// If w is complex, X and B are represented as NA x 2 matrices,
// the first column of each being the real part and the second
// being the imaginary part.
//
// "s" is a scaling factor (<= 1), computed by DLALN2, which is
// so chosen that X can be computed without overflow.  X is further
// scaled if necessary to assure that norm(ca A - w D)*norm(X) is less
// than overflow.
//
// If both singular values of (ca A - w D) are less than SMIN,
// SMIN*identity will be used instead of (ca A - w D).  If only one
// singular value is less than SMIN, one element of (ca A - w D) will be
// perturbed enough to make the smallest singular value roughly SMIN.
// If both singular values are at least SMIN, (ca A - w D) will not be
// perturbed.  In any case, the perturbation will be at most some small
// multiple of max( SMIN, ulp*norm(ca A - w D) ).  The singular values
// are computed by infinity-norm approximations, and thus will only be
// correct to a factor of 2 or so.
//
// Note: all input quantities are assumed to be smaller than overflow
// by a reasonable factor.  (See BIGNUM.)
func Dlaln2(ltrans bool, na, nw *int, smin, ca *float64, a *mat.Matrix, lda *int, d1, d2 *float64, b *mat.Matrix, ldb *int, wr, wi *float64, x *mat.Matrix, ldx *int, scale, xnorm *float64, info *int) {
	var bbnd, bi1, bi2, bignum, bnorm, br1, br2, ci21, ci22, cmax, cnorm, cr21, cr22, csi, csr, li21, lr21, one, smini, smlnum, temp, two, u22abs, ui11, ui11r, ui12, ui12s, ui22, ur11, ur11r, ur12, ur12s, ur22, xi1, xi2, xr1, xr2, zero float64
	var icmax, j int
	rswap := []bool{false, true, false, true}
	zswap := []bool{false, false, true, true}
	ipivot := make([]int, 4*4)

	ci := mf(2, 2, opts)
	cr := mf(2, 2, opts)

	zero = 0.0
	one = 1.0
	two = 2.0

	ipivot[0+(0)*4], ipivot[1+(0)*4], ipivot[2+(0)*4], ipivot[3+(0)*4], ipivot[0+(1)*4], ipivot[1+(1)*4], ipivot[2+(1)*4], ipivot[3+(1)*4], ipivot[0+(2)*4], ipivot[1+(2)*4], ipivot[2+(2)*4], ipivot[3+(2)*4], ipivot[0+(3)*4], ipivot[1+(3)*4], ipivot[2+(3)*4], ipivot[3+(3)*4] = 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1

	//     Compute BIGNUM
	smlnum = two * Dlamch(SafeMinimum)
	bignum = one / smlnum
	smini = math.Max(*smin, smlnum)

	//     Don't check for input errors
	(*info) = 0

	//     Standard Initializations
	(*scale) = one

	if (*na) == 1 {
		//        1 x 1  (i.e., scalar) system   C X = B
		if (*nw) == 1 {
			//           Real 1x1 system.
			//
			//           C = ca A - w D
			csr = (*ca)*a.Get(0, 0) - (*wr)*(*d1)
			cnorm = math.Abs(csr)

			//           If | C | < SMINI, use C = SMINI
			if cnorm < smini {
				csr = smini
				cnorm = smini
				(*info) = 1
			}

			//           Check scaling for  X = B / C
			bnorm = math.Abs(b.Get(0, 0))
			if cnorm < one && bnorm > one {
				if bnorm > bignum*cnorm {
					(*scale) = one / bnorm
				}
			}

			//           Compute X
			x.Set(0, 0, (b.Get(0, 0)*(*scale))/csr)
			(*xnorm) = math.Abs(x.Get(0, 0))
		} else {
			//           Complex 1x1 system (w is complex)
			//
			//           C = ca A - w D
			csr = (*ca)*a.Get(0, 0) - (*wr)*(*d1)
			csi = -(*wi) * (*d1)
			cnorm = math.Abs(csr) + math.Abs(csi)

			//           If | C | < SMINI, use C = SMINI
			if cnorm < smini {
				csr = smini
				csi = zero
				cnorm = smini
				(*info) = 1
			}

			//           Check scaling for  X = B / C
			bnorm = math.Abs(b.Get(0, 0)) + math.Abs(b.Get(0, 1))
			if cnorm < one && bnorm > one {
				if bnorm > bignum*cnorm {
					(*scale) = one / bnorm
				}
			}

			//           Compute X
			_scaleB00 := (*scale) * b.Get(0, 0)
			_scaleB01 := (*scale) * b.Get(0, 1)
			Dladiv(&_scaleB00, &_scaleB01, &csr, &csi, x.GetPtr(0, 0), x.GetPtr(0, 1))
			(*xnorm) = math.Abs(x.Get(0, 0)) + math.Abs(x.Get(0, 1))
		}

	} else {
		//        2x2 System
		//
		//        Compute the real part of  C = ca A - w D  (or  ca A**T - w D )
		cr.Set(0, 0, (*ca)*a.Get(0, 0)-(*wr)*(*d1))
		cr.Set(1, 1, (*ca)*a.Get(1, 1)-(*wr)*(*d2))
		if ltrans {
			cr.Set(0, 1, (*ca)*a.Get(1, 0))
			cr.Set(1, 0, (*ca)*a.Get(0, 1))
		} else {
			cr.Set(1, 0, (*ca)*a.Get(1, 0))
			cr.Set(0, 1, (*ca)*a.Get(0, 1))
		}

		if (*nw) == 1 {
			//           Real 2x2 system  (w is real)
			//
			//           Find the largest element in C
			cmax = zero
			icmax = 0

			for j = 1; j <= 4; j++ {
				if math.Abs(cr.GetIdx(j-1)) > cmax {
					cmax = math.Abs(cr.GetIdx(j - 1))
					icmax = j
				}
			}

			//           If norm(C) < SMINI, use SMINI*identity.
			if cmax < smini {
				bnorm = math.Max(math.Abs(b.Get(0, 0)), math.Abs(b.Get(1, 0)))
				if smini < one && bnorm > one {
					if bnorm > bignum*smini {
						(*scale) = one / bnorm
					}
				}
				temp = (*scale) / smini
				x.Set(0, 0, temp*b.Get(0, 0))
				x.Set(1, 0, temp*b.Get(1, 0))
				(*xnorm) = temp * bnorm
				(*info) = 1
				return
			}

			//           Gaussian elimination with complete pivoting.
			ur11 = cr.GetIdx(icmax - 1)
			cr21 = cr.GetIdx(ipivot[2-1+(icmax-1)*4] - 1)
			ur12 = cr.GetIdx(ipivot[3-1+(icmax-1)*4] - 1)
			cr22 = cr.GetIdx(ipivot[4-1+(icmax-1)*4] - 1)
			ur11r = one / ur11
			lr21 = ur11r * cr21
			ur22 = cr22 - ur12*lr21

			//           If smaller pivot < SMINI, use SMINI
			if math.Abs(ur22) < smini {
				ur22 = smini
				(*info) = 1
			}
			if rswap[icmax-1] {
				br1 = b.Get(1, 0)
				br2 = b.Get(0, 0)
			} else {
				br1 = b.Get(0, 0)
				br2 = b.Get(1, 0)
			}
			br2 = br2 - lr21*br1
			bbnd = math.Max(math.Abs(br1*(ur22*ur11r)), math.Abs(br2))
			if bbnd > one && math.Abs(ur22) < one {
				if bbnd >= bignum*math.Abs(ur22) {
					(*scale) = one / bbnd
				}
			}

			xr2 = (br2 * (*scale)) / ur22
			xr1 = ((*scale)*br1)*ur11r - xr2*(ur11r*ur12)
			if zswap[icmax-1] {
				x.Set(0, 0, xr2)
				x.Set(1, 0, xr1)
			} else {
				x.Set(0, 0, xr1)
				x.Set(1, 0, xr2)
			}
			(*xnorm) = math.Max(math.Abs(xr1), math.Abs(xr2))

			//           Further scaling if  norm(A) norm(X) > overflow
			if (*xnorm) > one && cmax > one {
				if (*xnorm) > bignum/cmax {
					temp = cmax / bignum
					x.Set(0, 0, temp*x.Get(0, 0))
					x.Set(1, 0, temp*x.Get(1, 0))
					(*xnorm) = temp * (*xnorm)
					(*scale) = temp * (*scale)
				}
			}
		} else {
			//           Complex 2x2 system  (w is complex)
			//
			//           Find the largest element in C
			ci.Set(0, 0, -(*wi)*(*d1))
			ci.Set(1, 0, zero)
			ci.Set(0, 1, zero)
			ci.Set(1, 1, -(*wi)*(*d2))
			cmax = zero
			icmax = 0

			for j = 1; j <= 4; j++ {
				if math.Abs(cr.GetIdx(j-1))+math.Abs(ci.GetIdx(j-1)) > cmax {
					cmax = math.Abs(cr.GetIdx(j-1)) + math.Abs(ci.GetIdx(j-1))
					icmax = j
				}
			}

			//           If norm(C) < SMINI, use SMINI*identity.
			if cmax < smini {
				bnorm = math.Max(math.Abs(b.Get(0, 0))+math.Abs(b.Get(0, 1)), math.Abs(b.Get(1, 0))+math.Abs(b.Get(1, 1)))
				if smini < one && bnorm > one {
					if bnorm > bignum*smini {
						(*scale) = one / bnorm
					}
				}
				temp = (*scale) / smini
				x.Set(0, 0, temp*b.Get(0, 0))
				x.Set(1, 0, temp*b.Get(1, 0))
				x.Set(0, 1, temp*b.Get(0, 1))
				x.Set(1, 1, temp*b.Get(1, 1))
				(*xnorm) = temp * bnorm
				(*info) = 1
				return
			}

			//           Gaussian elimination with complete pivoting.
			ur11 = cr.GetIdx(icmax - 1)
			ui11 = ci.GetIdx(icmax - 1)
			cr21 = cr.GetIdx(ipivot[2-1+(icmax-1)*4] - 1)
			ci21 = ci.GetIdx(ipivot[2-1+(icmax-1)*4] - 1)
			ur12 = cr.GetIdx(ipivot[3-1+(icmax-1)*4] - 1)
			ui12 = ci.GetIdx(ipivot[3-1+(icmax-1)*4] - 1)
			cr22 = cr.GetIdx(ipivot[4-1+(icmax-1)*4] - 1)
			ci22 = ci.GetIdx(ipivot[4-1+(icmax-1)*4] - 1)
			if icmax == 1 || icmax == 4 {
				//              Code when off-diagonals of pivoted C are real
				if math.Abs(ur11) > math.Abs(ui11) {
					temp = ui11 / ur11
					ur11r = one / (ur11 * (one + math.Pow(temp, 2)))
					ui11r = -temp * ur11r
				} else {
					temp = ur11 / ui11
					ui11r = -one / (ui11 * (one + math.Pow(temp, 2)))
					ur11r = -temp * ui11r
				}
				lr21 = cr21 * ur11r
				li21 = cr21 * ui11r
				ur12s = ur12 * ur11r
				ui12s = ur12 * ui11r
				ur22 = cr22 - ur12*lr21
				ui22 = ci22 - ur12*li21
			} else {
				//              Code when diagonals of pivoted C are real
				ur11r = one / ur11
				ui11r = zero
				lr21 = cr21 * ur11r
				li21 = ci21 * ur11r
				ur12s = ur12 * ur11r
				ui12s = ui12 * ur11r
				ur22 = cr22 - ur12*lr21 + ui12*li21
				ui22 = -ur12*li21 - ui12*lr21
			}
			u22abs = math.Abs(ur22) + math.Abs(ui22)

			//           If smaller pivot < SMINI, use SMINI
			if u22abs < smini {
				ur22 = smini
				ui22 = zero
				(*info) = 1
			}
			if rswap[icmax-1] {
				br2 = b.Get(0, 0)
				br1 = b.Get(1, 0)
				bi2 = b.Get(0, 1)
				bi1 = b.Get(1, 1)
			} else {
				br1 = b.Get(0, 0)
				br2 = b.Get(1, 0)
				bi1 = b.Get(0, 1)
				bi2 = b.Get(1, 1)
			}
			br2 = br2 - lr21*br1 + li21*bi1
			bi2 = bi2 - li21*br1 - lr21*bi1
			bbnd = math.Max((math.Abs(br1)+math.Abs(bi1))*(u22abs*(math.Abs(ur11r)+math.Abs(ui11r))), math.Abs(br2)+math.Abs(bi2))
			if bbnd > one && u22abs < one {
				if bbnd >= bignum*u22abs {
					(*scale) = one / bbnd
					br1 = (*scale) * br1
					bi1 = (*scale) * bi1
					br2 = (*scale) * br2
					bi2 = (*scale) * bi2
				}
			}

			Dladiv(&br2, &bi2, &ur22, &ui22, &xr2, &xi2)
			xr1 = ur11r*br1 - ui11r*bi1 - ur12s*xr2 + ui12s*xi2
			xi1 = ui11r*br1 + ur11r*bi1 - ui12s*xr2 - ur12s*xi2
			if zswap[icmax-1] {
				x.Set(0, 0, xr2)
				x.Set(1, 0, xr1)
				x.Set(0, 1, xi2)
				x.Set(1, 1, xi1)
			} else {
				x.Set(0, 0, xr1)
				x.Set(1, 0, xr2)
				x.Set(0, 1, xi1)
				x.Set(1, 1, xi2)
			}
			(*xnorm) = math.Max(math.Abs(xr1)+math.Abs(xi1), math.Abs(xr2)+math.Abs(xi2))

			//           Further scaling if  norm(A) norm(X) > overflow
			if (*xnorm) > one && cmax > one {
				if (*xnorm) > bignum/cmax {
					temp = cmax / bignum
					x.Set(0, 0, temp*x.Get(0, 0))
					x.Set(1, 0, temp*x.Get(1, 0))
					x.Set(0, 1, temp*x.Get(0, 1))
					x.Set(1, 1, temp*x.Get(1, 1))
					(*xnorm) = temp * (*xnorm)
					(*scale) = temp * (*scale)
				}
			}
		}
	}
}
