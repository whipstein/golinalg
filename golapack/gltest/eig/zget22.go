package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zget22 does an eigenvector check.
//
// The basic test is:
//
//    RESULT(1) = | A E  -  E W | / ( |A| |E| ulp )
//
// using the 1-norm.  It also tests the normalization of E:
//
//    RESULT(2) = maxint | m-norm(E(j)) - 1 | / ( n ulp )
//                 j
//
// where E(j) is the j-th eigenvector, and m-norm is the maxint-norm of a
// vector.  The maxint-norm of a complex n-vector x in this case is the
// maximum of |re(x(i)| + |im(x(i)| over i = 1, ..., n.
func Zget22(transa, transe, transw byte, n *int, a *mat.CMatrix, lda *int, e *mat.CMatrix, lde *int, w, work *mat.CVector, rwork, result *mat.Vector) {
	var norma, norme byte
	var cone, czero, wtemp complex128
	var anorm, enorm, enrmax, enrmin, errnrm, one, temp1, ulp, unfl, zero float64
	var itrnse, itrnsw, j, jcol, joff, jrow, jvec int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Initialize RESULT (in case N=0)
	result.Set(0, zero)
	result.Set(1, zero)
	if (*n) <= 0 {
		return
	}

	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Precision)

	itrnse = 0
	itrnsw = 0
	norma = 'O'
	norme = 'O'

	if transa == 'T' || transa == 'C' {
		norma = 'I'
	}

	if transe == 'T' {
		itrnse = 1
		norme = 'I'
	} else if transe == 'C' {
		itrnse = 2
		norme = 'I'
	}

	if transw == 'C' {
		itrnsw = 1
	}

	//     Normalization of E:
	enrmin = one / ulp
	enrmax = zero
	if itrnse == 0 {
		for jvec = 1; jvec <= (*n); jvec++ {
			temp1 = zero
			for j = 1; j <= (*n); j++ {
				temp1 = maxf64(temp1, math.Abs(e.GetRe(j-1, jvec-1))+math.Abs(e.GetIm(j-1, jvec-1)))
			}
			enrmin = minf64(enrmin, temp1)
			enrmax = maxf64(enrmax, temp1)
		}
	} else {
		for jvec = 1; jvec <= (*n); jvec++ {
			rwork.Set(jvec-1, zero)
		}

		for j = 1; j <= (*n); j++ {
			for jvec = 1; jvec <= (*n); jvec++ {
				rwork.Set(jvec-1, maxf64(rwork.Get(jvec-1), math.Abs(e.GetRe(jvec-1, j-1))+math.Abs(e.GetIm(jvec-1, j-1))))
			}
		}

		for jvec = 1; jvec <= (*n); jvec++ {
			enrmin = minf64(enrmin, rwork.Get(jvec-1))
			enrmax = maxf64(enrmax, rwork.Get(jvec-1))
		}
	}

	//     Norm of A:
	anorm = maxf64(golapack.Zlange(norma, n, n, a, lda, rwork), unfl)

	//     Norm of E:
	enorm = maxf64(golapack.Zlange(norme, n, n, e, lde, rwork), ulp)

	//     Norm of error:
	//
	//     Error =  AE - EW
	golapack.Zlaset('F', n, n, &czero, &czero, work.CMatrix(*n, opts), n)

	joff = 0
	for jcol = 1; jcol <= (*n); jcol++ {
		if itrnsw == 0 {
			wtemp = w.Get(jcol - 1)
		} else {
			wtemp = w.GetConj(jcol - 1)
		}

		if itrnse == 0 {
			for jrow = 1; jrow <= (*n); jrow++ {
				work.Set(joff+jrow-1, e.Get(jrow-1, jcol-1)*wtemp)
			}
		} else if itrnse == 1 {
			for jrow = 1; jrow <= (*n); jrow++ {
				work.Set(joff+jrow-1, e.Get(jcol-1, jrow-1)*wtemp)
			}
		} else {
			for jrow = 1; jrow <= (*n); jrow++ {
				work.Set(joff+jrow-1, e.GetConj(jcol-1, jrow-1)*wtemp)
			}
		}
		joff = joff + (*n)
	}

	err = goblas.Zgemm(mat.TransByte(transa), mat.TransByte(transe), *n, *n, *n, cone, a, *lda, e, *lde, -cone, work.CMatrix(*n, opts), *n)

	errnrm = golapack.Zlange('O', n, n, work.CMatrix(*n, opts), n, rwork) / enorm

	//     Compute RESULT(1) (avoiding under/overflow)
	if anorm > errnrm {
		result.Set(0, (errnrm/anorm)/ulp)
	} else {
		if anorm < one {
			result.Set(0, (minf64(errnrm, anorm)/anorm)/ulp)
		} else {
			result.Set(0, minf64(errnrm/anorm, one)/ulp)
		}
	}

	//     Compute RESULT(2) : the normalization error in E.
	result.Set(1, maxf64(math.Abs(enrmax-one), math.Abs(enrmin-one))/(float64(*n)*ulp))
}
