package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zget22 does an eigenvector check.
//
// The basic test is:
//
//    RESULT(1) = | A E  -  E W | / ( |A| |E| ulp )
//
// using the 1-norm.  It also tests the normalization of E:
//
//    RESULT(2) = max | m-norm(E(j)) - 1 | / ( n ulp )
//                 j
//
// where E(j) is the j-th eigenvector, and m-norm is the max-norm of a
// vector.  The max-norm of a complex n-vector x in this case is the
// maximum of |re(x(i)| + |im(x(i)| over i = 1, ..., n.
func zget22(transa, transe, transw mat.MatTrans, n int, a, e *mat.CMatrix, w, work *mat.CVector, rwork, result *mat.Vector) {
	var norma, norme byte
	var cone, czero, wtemp complex128
	var anorm, enorm, enrmax, enrmin, errnrm, one, temp1, ulp, unfl, zero float64
	var itrnse, itrnsw, j, jcol, joff, jrow, jvec int
	var err error

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Initialize RESULT (in case N=0)
	result.Set(0, zero)
	result.Set(1, zero)
	if n <= 0 {
		return
	}

	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Precision)

	itrnse = 0
	itrnsw = 0
	norma = 'O'
	norme = 'O'

	if transa.IsTrans() {
		norma = 'I'
	}

	if transe == Trans {
		itrnse = 1
		norme = 'I'
	} else if transe == ConjTrans {
		itrnse = 2
		norme = 'I'
	}

	if transw == ConjTrans {
		itrnsw = 1
	}

	//     Normalization of E:
	enrmin = one / ulp
	enrmax = zero
	if itrnse == 0 {
		for jvec = 1; jvec <= n; jvec++ {
			temp1 = zero
			for j = 1; j <= n; j++ {
				temp1 = math.Max(temp1, math.Abs(e.GetRe(j-1, jvec-1))+math.Abs(e.GetIm(j-1, jvec-1)))
			}
			enrmin = math.Min(enrmin, temp1)
			enrmax = math.Max(enrmax, temp1)
		}
	} else {
		for jvec = 1; jvec <= n; jvec++ {
			rwork.Set(jvec-1, zero)
		}

		for j = 1; j <= n; j++ {
			for jvec = 1; jvec <= n; jvec++ {
				rwork.Set(jvec-1, math.Max(rwork.Get(jvec-1), math.Abs(e.GetRe(jvec-1, j-1))+math.Abs(e.GetIm(jvec-1, j-1))))
			}
		}

		for jvec = 1; jvec <= n; jvec++ {
			enrmin = math.Min(enrmin, rwork.Get(jvec-1))
			enrmax = math.Max(enrmax, rwork.Get(jvec-1))
		}
	}

	//     Norm of A:
	anorm = math.Max(golapack.Zlange(norma, n, n, a, rwork), unfl)

	//     Norm of E:
	enorm = math.Max(golapack.Zlange(norme, n, n, e, rwork), ulp)

	//     Norm of error:
	//
	//     Error =  AE - EW
	golapack.Zlaset(Full, n, n, czero, czero, work.CMatrix(n, opts))

	joff = 0
	for jcol = 1; jcol <= n; jcol++ {
		if itrnsw == 0 {
			wtemp = w.Get(jcol - 1)
		} else {
			wtemp = w.GetConj(jcol - 1)
		}

		if itrnse == 0 {
			for jrow = 1; jrow <= n; jrow++ {
				work.Set(joff+jrow-1, e.Get(jrow-1, jcol-1)*wtemp)
			}
		} else if itrnse == 1 {
			for jrow = 1; jrow <= n; jrow++ {
				work.Set(joff+jrow-1, e.Get(jcol-1, jrow-1)*wtemp)
			}
		} else {
			for jrow = 1; jrow <= n; jrow++ {
				work.Set(joff+jrow-1, e.GetConj(jcol-1, jrow-1)*wtemp)
			}
		}
		joff = joff + n
	}

	if err = goblas.Zgemm(transa, transe, n, n, n, cone, a, e, -cone, work.CMatrix(n, opts)); err != nil {
		panic(err)
	}

	errnrm = golapack.Zlange('O', n, n, work.CMatrix(n, opts), rwork) / enorm

	//     Compute RESULT(1) (avoiding under/overflow)
	if anorm > errnrm {
		result.Set(0, (errnrm/anorm)/ulp)
	} else {
		if anorm < one {
			result.Set(0, (math.Min(errnrm, anorm)/anorm)/ulp)
		} else {
			result.Set(0, math.Min(errnrm/anorm, one)/ulp)
		}
	}

	//     Compute RESULT(2) : the normalization error in E.
	result.Set(1, math.Max(math.Abs(enrmax-one), math.Abs(enrmin-one))/(float64(n)*ulp))
}
