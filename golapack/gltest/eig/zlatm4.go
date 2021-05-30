package eig

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"math"
	"math/cmplx"
)

// Zlatm4 generates basic square matrices, which may later be
// multiplied by others in order to produce test matrices.  It is
// intended mainly to be used to test the generalized eigenvalue
// routines.
//
// It first generates the diagonal and (possibly) subdiagonal,
// according to the value of ITYPE, NZ1, NZ2, RSIGN, AMAGN, and RCOND.
// It then fills in the upper triangle with random numbers, if TRIANG is
// non-zero.
func Zlatm4(itype, n, nz1, nz2 *int, rsign bool, amagn, rcond, triang *float64, idist *int, iseed *[]int, a *mat.CMatrix, lda *int) {
	var cone, ctemp, czero complex128
	var alpha, one, zero float64
	var i, isdb, isde, jc, jd, jr, k, kbeg, kend, klen int

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	if (*n) <= 0 {
		return
	}
	golapack.Zlaset('F', n, n, &czero, &czero, a, lda)

	//     Insure a correct ISEED
	if ((*iseed)[3] % 2) != 1 {
		(*iseed)[3] = (*iseed)[3] + 1
	}

	//     Compute diagonal and subdiagonal according to ITYPE, NZ1, NZ2,
	//     and RCOND
	if (*itype) != 0 {
		if absint(*itype) >= 4 {
			kbeg = maxint(1, minint(*n, (*nz1)+1))
			kend = maxint(kbeg, minint(*n, (*n)-(*nz2)))
			klen = kend + 1 - kbeg
		} else {
			kbeg = 1
			kend = (*n)
			klen = (*n)
		}
		isdb = 1
		isde = 0
		switch absint(*itype) {
		case 1:
			goto label10
		case 2:
			goto label30
		case 3:
			goto label50
		case 4:
			goto label80
		case 5:
			goto label100
		case 6:
			goto label120
		case 7:
			goto label140
		case 8:
			goto label160
		case 9:
			goto label180
		case 10:
			goto label200
		}

		//        abs(ITYPE) = 1: Identity
	label10:
		;
		for jd = 1; jd <= (*n); jd++ {
			a.Set(jd-1, jd-1, cone)
		}
		goto label220

		//        abs(ITYPE) = 2: Transposed Jordan block
	label30:
		;
		for jd = 1; jd <= (*n)-1; jd++ {
			a.Set(jd+1-1, jd-1, cone)
		}
		isdb = 1
		isde = (*n) - 1
		goto label220

		//        abs(ITYPE) = 3: Transposed Jordan block, followed by the
		//                        identity.
	label50:
		;
		k = ((*n) - 1) / 2
		for jd = 1; jd <= k; jd++ {
			a.Set(jd+1-1, jd-1, cone)
		}
		isdb = 1
		isde = k
		for jd = k + 2; jd <= 2*k+1; jd++ {
			a.Set(jd-1, jd-1, cone)
		}
		goto label220

		//        abs(ITYPE) = 4: 1,...,k
	label80:
		;
		for jd = kbeg; jd <= kend; jd++ {
			a.SetRe(jd-1, jd-1, float64(jd-(*nz1)))
		}
		goto label220

		//        abs(ITYPE) = 5: One large D value:
	label100:
		;
		for jd = kbeg + 1; jd <= kend; jd++ {
			a.SetRe(jd-1, jd-1, *rcond)
		}
		a.Set(kbeg-1, kbeg-1, cone)
		goto label220

		//        abs(ITYPE) = 6: One small D value:
	label120:
		;
		for jd = kbeg; jd <= kend-1; jd++ {
			a.Set(jd-1, jd-1, cone)
		}
		a.SetRe(kend-1, kend-1, *rcond)
		goto label220

		//        abs(ITYPE) = 7: Exponentially distributed D values:
	label140:
		;
		a.Set(kbeg-1, kbeg-1, cone)
		if klen > 1 {
			alpha = math.Pow(*rcond, one/float64(klen-1))
			for i = 2; i <= klen; i++ {
				a.SetRe((*nz1)+i-1, (*nz1)+i-1, math.Pow(alpha, float64(i-1)))
			}
		}
		goto label220

		//        abs(ITYPE) = 8: Arithmetically distributed D values:
	label160:
		;
		a.Set(kbeg-1, kbeg-1, cone)
		if klen > 1 {
			alpha = (one - (*rcond)) / float64(klen-1)
			for i = 2; i <= klen; i++ {
				a.SetRe((*nz1)+i-1, (*nz1)+i-1, float64(klen-i)*alpha+(*rcond))
			}
		}
		goto label220

		//        abs(ITYPE) = 9: Randomly distributed D values on ( RCOND, 1):
	label180:
		;
		alpha = math.Log(*rcond)
		for jd = kbeg; jd <= kend; jd++ {
			a.SetRe(jd-1, jd-1, math.Exp(alpha*matgen.Dlaran(iseed)))
		}
		goto label220

		//        abs(ITYPE) = 10: Randomly distributed D values from DIST
	label200:
		;
		for jd = kbeg; jd <= kend; jd++ {
			a.Set(jd-1, jd-1, matgen.Zlarnd(idist, iseed))
		}

	label220:
		;

		//        Scale by AMAGN
		for jd = kbeg; jd <= kend; jd++ {
			a.SetRe(jd-1, jd-1, (*amagn)*a.GetRe(jd-1, jd-1))
		}
		for jd = isdb; jd <= isde; jd++ {
			a.SetRe(jd+1-1, jd-1, (*amagn)*a.GetRe(jd+1-1, jd-1))
		}

		//        If RSIGN = .TRUE., assign random signs to diagonal and
		//        subdiagonal
		if rsign {
			for jd = kbeg; jd <= kend; jd++ {
				if a.GetRe(jd-1, jd-1) != zero {
					ctemp = matgen.Zlarnd(func() *int { y := 3; return &y }(), iseed)
					ctemp = ctemp / complex(cmplx.Abs(ctemp), 0)
					a.Set(jd-1, jd-1, ctemp*a.GetReCmplx(jd-1, jd-1))
				}
			}
			for jd = isdb; jd <= isde; jd++ {
				if a.GetRe(jd+1-1, jd-1) != zero {
					ctemp = matgen.Zlarnd(func() *int { y := 3; return &y }(), iseed)
					ctemp = ctemp / complex(cmplx.Abs(ctemp), 0)
					a.Set(jd+1-1, jd-1, ctemp*a.GetReCmplx(jd+1-1, jd-1))
				}
			}
		}

		//        Reverse if ITYPE < 0
		if (*itype) < 0 {
			for jd = kbeg; jd <= (kbeg+kend-1)/2; jd++ {
				ctemp = a.Get(jd-1, jd-1)
				a.Set(jd-1, jd-1, a.Get(kbeg+kend-jd-1, kbeg+kend-jd-1))
				a.Set(kbeg+kend-jd-1, kbeg+kend-jd-1, ctemp)
			}
			for jd = 1; jd <= ((*n)-1)/2; jd++ {
				ctemp = a.Get(jd+1-1, jd-1)
				a.Set(jd+1-1, jd-1, a.Get((*n)+1-jd-1, (*n)-jd-1))
				a.Set((*n)+1-jd-1, (*n)-jd-1, ctemp)
			}
		}

	}

	//     Fill in upper triangle
	if (*triang) != zero {
		for jc = 2; jc <= (*n); jc++ {
			for jr = 1; jr <= jc-1; jr++ {
				a.Set(jr-1, jc-1, complex(*triang, 0)*matgen.Zlarnd(idist, iseed))
			}
		}
	}
}
