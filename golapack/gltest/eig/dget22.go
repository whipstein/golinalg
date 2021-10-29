package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dget22 does an eigenvector check.
//
// The basic test is:
//
//    RESULT(1) = | A E  -  E W | / ( |A| |E| ulp )
//
// using the 1-norm.  It also tests the normalization of E:
//
//    RESULT(2) = math.Max | m-norm(E(j)) - 1 | / ( n ulp )
//                 j
//
// where E(j) is the j-th eigenvector, and m-norm is the math.Max-norm of a
// vector.  If an eigenvector is complex, as determined from WI(j)
// nonzero, then the math.Max-norm of the vector ( er + i*ei ) is the maximum
// of
//    |er(1)| + |ei(1)|, ... , |er(n)| + |ei(n)|
//
// W is a block diagonal matrix, with a 1 by 1 block for each real
// eigenvalue and a 2 by 2 block for each complex conjugate pair.
// If eigenvalues j and j+1 are a complex conjugate pair, so that
// WR(j) = WR(j+1) = wr and WI(j) = - WI(j+1) = wi, then the 2 by 2
// block corresponding to the pair will be:
//
//    (  wr  wi  )
//    ( -wi  wr  )
//
// Such a block multiplying an n by 2 matrix ( ur ui ) on the right
// will be the same as multiplying  ur + i*ui  by  wr + i*wi.
//
// To handle various schemes for storage of left eigenvectors, there are
// options to use A-transpose instead of A, E-transpose instead of E,
// and/or W-transpose instead of W.
func dget22(transa, transe, transw mat.MatTrans, n int, a, e *mat.Matrix, wr, wi, work, result *mat.Vector) {
	var norma, norme byte
	var anorm, enorm, enrmax, enrmin, errnrm, one, temp1, ulp, unfl, zero float64
	var iecol, ierow, ince, ipair, itrnse, j, jcol, jvec int
	var err error

	wmat := mf(2, 2, opts)

	zero = 0.0
	one = 1.0

	//     Initialize RESULT (in case N=0)
	result.Set(0, zero)
	result.Set(1, zero)
	if n <= 0 {
		return
	}

	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Precision)

	itrnse = 0
	ince = 1
	norma = 'O'
	norme = 'O'

	if transa.IsTrans() {
		norma = 'I'
	}
	if transe.IsTrans() {
		norme = 'I'
		itrnse = 1
		ince = e.Rows
	}

	//     Check normalization of E
	enrmin = one / ulp
	enrmax = zero
	if itrnse == 0 {
		//        Eigenvectors are column vectors.
		ipair = 0
		for jvec = 1; jvec <= n; jvec++ {
			temp1 = zero
			if ipair == 0 && jvec < n && wi.Get(jvec-1) != zero {
				ipair = 1
			}
			if ipair == 1 {
				//              Complex eigenvector
				for j = 1; j <= n; j++ {
					temp1 = math.Max(temp1, math.Abs(e.Get(j-1, jvec-1))+math.Abs(e.Get(j-1, jvec)))
				}
				enrmin = math.Min(enrmin, temp1)
				enrmax = math.Max(enrmax, temp1)
				ipair = 2
			} else if ipair == 2 {
				ipair = 0
			} else {
				//              Real eigenvector
				for j = 1; j <= n; j++ {
					temp1 = math.Max(temp1, math.Abs(e.Get(j-1, jvec-1)))
				}
				enrmin = math.Min(enrmin, temp1)
				enrmax = math.Max(enrmax, temp1)
				ipair = 0
			}
		}

	} else {
		//        Eigenvectors are row vectors.
		for jvec = 1; jvec <= n; jvec++ {
			work.Set(jvec-1, zero)
		}

		for j = 1; j <= n; j++ {
			ipair = 0
			for jvec = 1; jvec <= n; jvec++ {
				if ipair == 0 && jvec < n && wi.Get(jvec-1) != zero {
					ipair = 1
				}
				if ipair == 1 {
					work.Set(jvec-1, math.Max(work.Get(jvec-1), math.Abs(e.Get(j-1, jvec-1))+math.Abs(e.Get(j-1, jvec))))
					work.Set(jvec, work.Get(jvec-1))
				} else if ipair == 2 {
					ipair = 0
				} else {
					work.Set(jvec-1, math.Max(work.Get(jvec-1), math.Abs(e.Get(j-1, jvec-1))))
					ipair = 0
				}
			}
		}

		for jvec = 1; jvec <= n; jvec++ {
			enrmin = math.Min(enrmin, work.Get(jvec-1))
			enrmax = math.Max(enrmax, work.Get(jvec-1))
		}
	}

	//     Norm of A:
	anorm = math.Max(golapack.Dlange(norma, n, n, a, work), unfl)

	//     Norm of E:
	enorm = math.Max(golapack.Dlange(norme, n, n, e, work), ulp)

	//     Norm of error:
	//
	//     Error =  AE - EW
	golapack.Dlaset(Full, n, n, zero, zero, work.Matrix(n, opts))

	ipair = 0
	ierow = 1
	iecol = 1

	for jcol = 1; jcol <= n; jcol++ {
		if itrnse == 1 {
			ierow = jcol
		} else {
			iecol = jcol
		}

		if ipair == 0 && wi.Get(jcol-1) != zero {
			ipair = 1
		}

		if ipair == 1 {
			wmat.Set(0, 0, wr.Get(jcol-1))
			wmat.Set(1, 0, -wi.Get(jcol-1))
			wmat.Set(0, 1, wi.Get(jcol-1))
			wmat.Set(1, 1, wr.Get(jcol-1))
			if err = goblas.Dgemm(transe, transw, n, 2, 2, one, e.Off(ierow-1, iecol-1), wmat, zero, work.MatrixOff(n*(jcol-1), n, opts)); err != nil {
				panic(err)
			}
			ipair = 2
		} else if ipair == 2 {
			ipair = 0

		} else {

			goblas.Daxpy(n, wr.Get(jcol-1), e.Vector(ierow-1, iecol-1, ince), work.Off(n*(jcol-1), 1))
			ipair = 0
		}

	}

	if err = goblas.Dgemm(transa, transe, n, n, n, one, a, e, -one, work.Matrix(n, opts)); err != nil {
		panic(err)
	}

	errnrm = golapack.Dlange('O', n, n, work.Matrix(n, opts), work.Off(n*n)) / enorm

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
