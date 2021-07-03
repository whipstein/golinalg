package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dget52 does an eigenvector check for the generalized eigenvalue
// problem.
//
// The basic test for right eigenvectors is:
//
//                           | b(j) A E(j) -  a(j) B E(j) |
//         RESULT(1) = maxf64   -------------------------------
//                      j    n ulp maxf64( |b(j) A|, |a(j) B| )
//
// using the 1-norm.  Here, a(j)/b(j) = w is the j-th generalized
// eigenvalue of A - w B, or, equivalently, b(j)/a(j) = m is the j-th
// generalized eigenvalue of m A - B.
//
// For real eigenvalues, the test is straightforward.  For complex
// eigenvalues, E(j) and a(j) are complex, represented by
// Er(j) + i*Ei(j) and ar(j) + i*ai(j), resp., so the test for that
// eigenvector becomes
//
//                 maxf64( |Wr|, |Wi| )
//     --------------------------------------------
//     n ulp maxf64( |b(j) A|, (|ar(j)|+|ai(j)|) |B| )
//
// where
//
//     Wr = b(j) A Er(j) - ar(j) B Er(j) + ai(j) B Ei(j)
//
//     Wi = b(j) A Ei(j) - ai(j) B Er(j) - ar(j) B Ei(j)
//
//                         T   T  _
// For left eigenvectors, A , B , a, and b  are used.
//
// DGET52 also tests the normalization of E.  Each eigenvector is
// supposed to be normalized so that the maximum "absolute value"
// of its elements is 1, where in this case, "absolute value"
// of a complex value x is  |Re(x)| + |Im(x)| ; let us call this
// maximum "absolute value" norm of a vector v  M(v).
// if a(j)=b(j)=0, then the eigenvector is set to be the jth coordinate
// vector.  The normalization test is:
//
//         RESULT(2) =      maxf64       | M(v(j)) - 1 | / ( n ulp )
//                    eigenvectors v(j)
func Dget52(left bool, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, e *mat.Matrix, lde *int, alphar, alphai, beta, work, result *mat.Vector) {
	var ilcplx bool
	var normab, trans byte
	var abmax, acoef, alfmax, anorm, bcoefi, bcoefr, betmax, bnorm, enorm, enrmer, errnrm, one, safmax, safmin, salfi, salfr, sbeta, scale, temp1, ten, ulp, zero float64
	var j, jvec int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	ten = 10.0

	result.Set(0, zero)
	result.Set(1, zero)
	if (*n) <= 0 {
		return
	}

	safmin = golapack.Dlamch(SafeMinimum)
	safmax = one / safmin
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	if left {
		trans = 'T'
		normab = 'I'
	} else {
		trans = 'N'
		normab = 'O'
	}

	//     Norm of A, B, and E:
	anorm = maxf64(golapack.Dlange(normab, n, n, a, lda, work), safmin)
	bnorm = maxf64(golapack.Dlange(normab, n, n, b, ldb, work), safmin)
	enorm = maxf64(golapack.Dlange('O', n, n, e, lde, work), ulp)
	alfmax = safmax / maxf64(one, bnorm)
	betmax = safmax / maxf64(one, anorm)

	//     Compute error matrix.
	//     Column i = ( b(i) A - a(i) B ) E(i) / maxf64( |a(i) B| |b(i) A| )
	ilcplx = false
	for jvec = 1; jvec <= (*n); jvec++ {
		if ilcplx {
			//           2nd Eigenvalue/-vector of pair -- do nothing
			ilcplx = false
		} else {
			salfr = alphar.Get(jvec - 1)
			salfi = alphai.Get(jvec - 1)
			sbeta = beta.Get(jvec - 1)
			if salfi == zero {
				//              Real eigenvalue and -vector
				abmax = maxf64(math.Abs(salfr), math.Abs(sbeta))
				if math.Abs(salfr) > alfmax || math.Abs(sbeta) > betmax || abmax < one {
					scale = one / maxf64(abmax, safmin)
					salfr = scale * salfr
					sbeta = scale * sbeta
				}
				scale = one / maxf64(math.Abs(salfr)*bnorm, math.Abs(sbeta)*anorm, safmin)
				acoef = scale * sbeta
				bcoefr = scale * salfr
				err = goblas.Dgemv(mat.TransByte(trans), *n, *n, acoef, a, *lda, e.Vector(0, jvec-1), 1, zero, work.Off((*n)*(jvec-1)+1-1), 1)
				err = goblas.Dgemv(mat.TransByte(trans), *n, *n, -bcoefr, b, *lda, e.Vector(0, jvec-1), 1, one, work.Off((*n)*(jvec-1)+1-1), 1)
			} else {
				//              Complex conjugate pair
				ilcplx = true
				if jvec == (*n) {
					result.Set(0, ten/ulp)
					return
				}
				abmax = maxf64(math.Abs(salfr)+math.Abs(salfi), math.Abs(sbeta))
				if math.Abs(salfr)+math.Abs(salfi) > alfmax || math.Abs(sbeta) > betmax || abmax < one {
					scale = one / maxf64(abmax, safmin)
					salfr = scale * salfr
					salfi = scale * salfi
					sbeta = scale * sbeta
				}
				scale = one / maxf64((math.Abs(salfr)+math.Abs(salfi))*bnorm, math.Abs(sbeta)*anorm, safmin)
				acoef = scale * sbeta
				bcoefr = scale * salfr
				bcoefi = scale * salfi
				if left {
					bcoefi = -bcoefi
				}

				err = goblas.Dgemv(mat.TransByte(trans), *n, *n, acoef, a, *lda, e.Vector(0, jvec-1), 1, zero, work.Off((*n)*(jvec-1)+1-1), 1)
				err = goblas.Dgemv(mat.TransByte(trans), *n, *n, -bcoefr, b, *lda, e.Vector(0, jvec-1), 1, one, work.Off((*n)*(jvec-1)+1-1), 1)
				err = goblas.Dgemv(mat.TransByte(trans), *n, *n, bcoefi, b, *lda, e.Vector(0, jvec+1-1), 1, one, work.Off((*n)*(jvec-1)+1-1), 1)

				err = goblas.Dgemv(mat.TransByte(trans), *n, *n, acoef, a, *lda, e.Vector(0, jvec+1-1), 1, zero, work.Off((*n)*jvec+1-1), 1)
				err = goblas.Dgemv(mat.TransByte(trans), *n, *n, -bcoefi, b, *lda, e.Vector(0, jvec-1), 1, one, work.Off((*n)*jvec+1-1), 1)
				err = goblas.Dgemv(mat.TransByte(trans), *n, *n, -bcoefr, b, *lda, e.Vector(0, jvec+1-1), 1, one, work.Off((*n)*jvec+1-1), 1)
			}
		}
	}

	errnrm = golapack.Dlange('O', n, n, work.Matrix(*n, opts), n, work.Off(int(math.Pow(float64(*n), 2))+1-1)) / enorm

	//     Compute RESULT(1)
	result.Set(0, errnrm/ulp)

	//     Normalization of E:
	enrmer = zero
	ilcplx = false
	for jvec = 1; jvec <= (*n); jvec++ {
		if ilcplx {
			ilcplx = false
		} else {
			temp1 = zero
			if alphai.Get(jvec-1) == zero {
				for j = 1; j <= (*n); j++ {
					temp1 = maxf64(temp1, math.Abs(e.Get(j-1, jvec-1)))
				}
				enrmer = maxf64(enrmer, temp1-one)
			} else {
				ilcplx = true
				for j = 1; j <= (*n); j++ {
					temp1 = maxf64(temp1, math.Abs(e.Get(j-1, jvec-1))+math.Abs(e.Get(j-1, jvec+1-1)))
				}
				enrmer = maxf64(enrmer, temp1-one)
			}
		}
	}

	//     Compute RESULT(2) : the normalization error in E.
	result.Set(1, enrmer/(float64(*n)*ulp))
}
