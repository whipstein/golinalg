package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Zstt21 checks a decomposition of the form
//
//    A = U S U**H
//
// where **H means conjugate transpose, A is real symmetric tridiagonal,
// U is unitary, and S is real and diagonal (if KBAND=0) or symmetric
// tridiagonal (if KBAND=1).  Two tests are performed:
//
//    RESULT(1) = | A - U S U**H | / ( |A| n ulp )
//
//    RESULT(2) = | I - U U**H | / ( n ulp )
func Zstt21(n, kband *int, ad, ae, sd, se *mat.Vector, u *mat.CMatrix, ldu *int, work *mat.CVector, rwork, result *mat.Vector) {
	var cone, czero complex128
	var anorm, one, temp1, temp2, ulp, unfl, wnorm, zero float64
	var j int

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     1)      Constants
	result.Set(0, zero)
	result.Set(1, zero)
	if (*n) <= 0 {
		return
	}

	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Precision)

	//     Do Test 1
	//
	//     Copy A & Compute its 1-Norm:
	golapack.Zlaset('F', n, n, &czero, &czero, work.CMatrix(*n, opts), n)

	anorm = zero
	temp1 = zero

	for j = 1; j <= (*n)-1; j++ {
		work.Set(((*n)+1)*(j-1)+1-1, ad.GetCmplx(j-1))
		work.Set(((*n)+1)*(j-1)+2-1, ae.GetCmplx(j-1))
		temp2 = ae.GetMag(j - 1)
		anorm = maxf64(anorm, ad.GetMag(j-1)+temp1+temp2)
		temp1 = temp2
	}

	work.Set(powint(*n, 2)-1, ad.GetCmplx((*n)-1))
	anorm = maxf64(anorm, ad.GetMag((*n)-1)+temp1, unfl)

	//     Norm of A - USU*
	for j = 1; j <= (*n); j++ {
		goblas.Zher(Lower, n, toPtrf64(-sd.Get(j-1)), u.CVector(0, j-1), func() *int { y := 1; return &y }(), work.CMatrix(*n, opts), n)
	}

	if (*n) > 1 && (*kband) == 1 {
		for j = 1; j <= (*n)-1; j++ {
			goblas.Zher2(Lower, n, toPtrc128(-se.GetCmplx(j-1)), u.CVector(0, j-1), func() *int { y := 1; return &y }(), u.CVector(0, j+1-1), func() *int { y := 1; return &y }(), work.CMatrix(*n, opts), n)
		}
	}

	wnorm = golapack.Zlanhe('1', 'L', n, work.CMatrix(*n, opts), n, rwork)

	if anorm > wnorm {
		result.Set(0, (wnorm/anorm)/(float64(*n)*ulp))
	} else {
		if anorm < one {
			result.Set(0, (minf64(wnorm, float64(*n)*anorm)/anorm)/(float64(*n)*ulp))
		} else {
			result.Set(0, minf64(wnorm/anorm, float64(*n))/(float64(*n)*ulp))
		}
	}

	//     Do Test 2
	//
	//     Compute  U U**H - I
	goblas.Zgemm(NoTrans, ConjTrans, n, n, n, &cone, u, ldu, u, ldu, &czero, work.CMatrix(*n, opts), n)

	for j = 1; j <= (*n); j++ {
		work.Set(((*n)+1)*(j-1)+1-1, work.Get(((*n)+1)*(j-1)+1-1)-cone)
	}

	result.Set(1, minf64(float64(*n), golapack.Zlange('1', n, n, work.CMatrix(*n, opts), n, rwork))/(float64(*n)*ulp))
}