package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
	"math/cmplx"
)

// Zget52 does an eigenvector check for the generalized eigenvalue
// problem.
//
// The basic test for right eigenvectors is:
//
//                           | b(i) A E(i) -  a(i) B E(i) |
//         RESULT(1) = maxint   -------------------------------
//                      i    n ulp maxint( |b(i) A|, |a(i) B| )
//
// using the 1-norm.  Here, a(i)/b(i) = w is the i-th generalized
// eigenvalue of A - w B, or, equivalently, b(i)/a(i) = m is the i-th
// generalized eigenvalue of m A - B.
//
//                         H   H  _      _
// For left eigenvectors, A , B , a, and b  are used.
//
// ZGET52 also tests the normalization of E.  Each eigenvector is
// supposed to be normalized so that the maximum "absolute value"
// of its elements is 1, where in this case, "absolute value"
// of a complex value x is  |Re(x)| + |Im(x)| ; let us call this
// maximum "absolute value" norm of a vector v  M(v).
// If a(i)=b(i)=0, then the eigenvector is set to be the jth coordinate
// vector. The normalization test is:
//
//         RESULT(2) =      maxint       | M(v(i)) - 1 | / ( n ulp )
//                    eigenvectors v(i)
func Zget52(left bool, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, e *mat.CMatrix, lde *int, alpha, beta, work *mat.CVector, rwork, result *mat.Vector) {
	var normab, trans byte
	var acoeff, alphai, bcoeff, betai, cone, czero complex128
	var abmax, alfmax, anorm, betmax, bnorm, enorm, enrmer, errnrm, one, safmax, safmin, scale, temp1, ulp, zero float64
	var j, jvec int

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	result.Set(0, zero)
	result.Set(1, zero)
	if (*n) <= 0 {
		return
	}

	safmin = golapack.Dlamch(SafeMinimum)
	safmax = one / safmin
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	if left {
		trans = 'C'
		normab = 'I'
	} else {
		trans = 'N'
		normab = 'O'
	}

	//     Norm of A, B, and E:
	anorm = maxf64(golapack.Zlange(normab, n, n, a, lda, rwork), safmin)
	bnorm = maxf64(golapack.Zlange(normab, n, n, b, ldb, rwork), safmin)
	enorm = maxf64(golapack.Zlange('O', n, n, e, lde, rwork), ulp)
	alfmax = safmax / maxf64(one, bnorm)
	betmax = safmax / maxf64(one, anorm)

	//     Compute error matrix.
	//     Column i = ( b(i) A - a(i) B ) E(i) / maxint( |a(i) B| |b(i) A| )
	for jvec = 1; jvec <= (*n); jvec++ {
		alphai = alpha.Get(jvec - 1)
		betai = beta.Get(jvec - 1)
		abmax = maxf64(abs1(alphai), abs1(betai))
		if abs1(alphai) > alfmax || abs1(betai) > betmax || abmax < one {
			scale = one / maxf64(abmax, safmin)
			alphai = complex(scale, 0) * alphai
			betai = complex(scale, 0) * betai
		}
		scale = one / maxf64(abs1(alphai)*bnorm, abs1(betai)*anorm, safmin)
		acoeff = complex(scale, 0) * betai
		bcoeff = complex(scale, 0) * alphai
		if left {
			acoeff = cmplx.Conj(acoeff)
			bcoeff = cmplx.Conj(bcoeff)
		}
		goblas.Zgemv(mat.TransByte(trans), n, n, &acoeff, a, lda, e.CVector(0, jvec-1), func() *int { y := 1; return &y }(), &czero, work.Off((*n)*(jvec-1)+1-1), func() *int { y := 1; return &y }())
		goblas.Zgemv(mat.TransByte(trans), n, n, toPtrc128(-bcoeff), b, lda, e.CVector(0, jvec-1), func() *int { y := 1; return &y }(), &cone, work.Off((*n)*(jvec-1)+1-1), func() *int { y := 1; return &y }())
	}

	errnrm = golapack.Zlange('O', n, n, work.CMatrix(*n, opts), n, rwork) / enorm

	//     Compute RESULT(1)
	result.Set(0, errnrm/ulp)

	//     Normalization of E:
	enrmer = zero
	for jvec = 1; jvec <= (*n); jvec++ {
		temp1 = zero
		for j = 1; j <= (*n); j++ {
			temp1 = maxf64(temp1, abs1(e.Get(j-1, jvec-1)))
		}
		enrmer = maxf64(enrmer, temp1-one)
	}

	//     Compute RESULT(2) : the normalization error in E.
	result.Set(1, enrmer/(float64(*n)*ulp))
}