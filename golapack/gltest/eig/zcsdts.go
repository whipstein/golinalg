package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zcsdts tests ZUNCSD, which, given an M-by-M partitioned unitary
// matrix X,
//              Q  M-Q
//       X = [ X11 X12 ] P   ,
//           [ X21 X22 ] M-P
//
// computes the CSD
//
//       [ U1    ]**T * [ X11 X12 ] * [ V1    ]
//       [    U2 ]      [ X21 X22 ]   [    V2 ]
//
//                             [  I  0  0 |  0  0  0 ]
//                             [  0  C  0 |  0 -S  0 ]
//                             [  0  0  0 |  0  0 -I ]
//                           = [---------------------] = [ D11 D12 ] .
//                             [  0  0  0 |  I  0  0 ]   [ D21 D22 ]
//                             [  0  S  0 |  0  C  0 ]
//                             [  0  0  I |  0  0  0 ]
//
// and also SORCSD2BY1, which, given
//          Q
//       [ X11 ] P   ,
//       [ X21 ] M-P
//
// computes the 2-by-1 CSD
//
//                                     [  I  0  0 ]
//                                     [  0  C  0 ]
//                                     [  0  0  0 ]
//       [ U1    ]**T * [ X11 ] * V1 = [----------] = [ D11 ] ,
//       [    U2 ]      [ X21 ]        [  0  0  0 ]   [ D21 ]
//                                     [  0  S  0 ]
//                                     [  0  0  I ]
func Zcsdts(m, p, q *int, x, xf *mat.CMatrix, ldx *int, u1 *mat.CMatrix, ldu1 *int, u2 *mat.CMatrix, ldu2 *int, v1t *mat.CMatrix, ldv1t *int, v2t *mat.CMatrix, ldv2t *int, theta *mat.Vector, iwork *[]int, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var one, zero complex128
	var eps2, piover2, realone, realzero, resid, ulp, ulpinv float64
	var i, info, r int
	var err error
	_ = err

	piover2 = 1.57079632679489662
	realone = 1.0
	realzero = 0.0
	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	ulp = golapack.Dlamch(Precision)
	ulpinv = realone / ulp

	//     The first half of the routine checks the 2-by-2 CSD
	golapack.Zlaset('F', m, m, &zero, &one, work.CMatrix(*ldx, opts), ldx)
	err = goblas.Zherk(Upper, ConjTrans, *m, *m, -realone, x, *ldx, realone, work.CMatrix(*ldx, opts), *ldx)
	if (*m) > 0 {
		eps2 = maxf64(ulp, golapack.Zlange('1', m, m, work.CMatrix(*ldx, opts), ldx, rwork)/float64(*m))
	} else {
		eps2 = ulp
	}
	r = minint(*p, (*m)-(*p), *q, (*m)-(*q))

	//     Copy the matrix X to the array XF.
	golapack.Zlacpy('F', m, m, x, ldx, xf, ldx)

	//     Compute the CSD
	golapack.Zuncsd('Y', 'Y', 'Y', 'Y', 'N', 'D', m, p, q, xf.Off(0, 0), ldx, xf.Off(0, (*q)+1-1), ldx, xf.Off((*p)+1-1, 0), ldx, xf.Off((*p)+1-1, (*q)+1-1), ldx, theta, u1, ldu1, u2, ldu2, v1t, ldv1t, v2t, ldv2t, work, lwork, rwork, toPtr(17*(r+2)), iwork, &info)

	//     Compute XF := diag(U1,U2)'*X*diag(V1,V2) - [D11 D12; D21 D22]
	golapack.Zlacpy('F', m, m, x, ldx, xf, ldx)

	err = goblas.Zgemm(NoTrans, ConjTrans, *p, *q, *q, one, xf, *ldx, v1t, *ldv1t, zero, work.CMatrix(*ldx, opts), *ldx)

	err = goblas.Zgemm(ConjTrans, NoTrans, *p, *q, *p, one, u1, *ldu1, work.CMatrix(*ldx, opts), *ldx, zero, xf, *ldx)

	for i = 1; i <= minint(*p, *q)-r; i++ {
		xf.Set(i-1, i-1, xf.Get(i-1, i-1)-one)
	}
	for i = 1; i <= r; i++ {
		xf.Set(minint(*p, *q)-r+i-1, minint(*p, *q)-r+i-1, xf.Get(minint(*p, *q)-r+i-1, minint(*p, *q)-r+i-1)-toCmplx(math.Cos(theta.Get(i-1))))
	}

	err = goblas.Zgemm(NoTrans, ConjTrans, *p, (*m)-(*q), (*m)-(*q), one, xf.Off(0, (*q)+1-1), *ldx, v2t, *ldv2t, zero, work.CMatrix(*ldx, opts), *ldx)

	err = goblas.Zgemm(ConjTrans, NoTrans, *p, (*m)-(*q), *p, one, u1, *ldu1, work.CMatrix(*ldx, opts), *ldx, zero, xf.Off(0, (*q)+1-1), *ldx)

	for i = 1; i <= minint(*p, (*m)-(*q))-r; i++ {
		xf.Set((*p)-i+1-1, (*m)-i+1-1, xf.Get((*p)-i+1-1, (*m)-i+1-1)+one)
	}
	for i = 1; i <= r; i++ {
		xf.Set((*p)-(minint(*p, (*m)-(*q))-r)+1-i-1, (*m)-(minint(*p, (*m)-(*q))-r)+1-i-1, xf.Get((*p)-(minint(*p, (*m)-(*q))-r)+1-i-1, (*m)-(minint(*p, (*m)-(*q))-r)+1-i-1)+toCmplx(math.Sin(theta.Get(r-i+1-1))))
	}

	err = goblas.Zgemm(NoTrans, ConjTrans, (*m)-(*p), *q, *q, one, xf.Off((*p)+1-1, 0), *ldx, v1t, *ldv1t, zero, work.CMatrix(*ldx, opts), *ldx)

	err = goblas.Zgemm(ConjTrans, NoTrans, (*m)-(*p), *q, (*m)-(*p), one, u2, *ldu2, work.CMatrix(*ldx, opts), *ldx, zero, xf.Off((*p)+1-1, 0), *ldx)

	for i = 1; i <= minint((*m)-(*p), *q)-r; i++ {
		xf.Set((*m)-i+1-1, (*q)-i+1-1, xf.Get((*m)-i+1-1, (*q)-i+1-1)-one)
	}
	for i = 1; i <= r; i++ {
		xf.Set((*m)-(minint((*m)-(*p), *q)-r)+1-i-1, (*q)-(minint((*m)-(*p), *q)-r)+1-i-1, xf.Get((*m)-(minint((*m)-(*p), *q)-r)+1-i-1, (*q)-(minint((*m)-(*p), *q)-r)+1-i-1)-toCmplx(math.Sin(theta.Get(r-i+1-1))))
	}

	err = goblas.Zgemm(NoTrans, ConjTrans, (*m)-(*p), (*m)-(*q), (*m)-(*q), one, xf.Off((*p)+1-1, (*q)+1-1), *ldx, v2t, *ldv2t, zero, work.CMatrix(*ldx, opts), *ldx)

	err = goblas.Zgemm(ConjTrans, NoTrans, (*m)-(*p), (*m)-(*q), (*m)-(*p), one, u2, *ldu2, work.CMatrix(*ldx, opts), *ldx, zero, xf.Off((*p)+1-1, (*q)+1-1), *ldx)

	for i = 1; i <= minint((*m)-(*p), (*m)-(*q))-r; i++ {
		xf.Set((*p)+i-1, (*q)+i-1, xf.Get((*p)+i-1, (*q)+i-1)-one)
	}
	for i = 1; i <= r; i++ {
		xf.Set((*p)+(minint((*m)-(*p), (*m)-(*q))-r)+i-1, (*q)+(minint((*m)-(*p), (*m)-(*q))-r)+i-1, xf.Get((*p)+(minint((*m)-(*p), (*m)-(*q))-r)+i-1, (*q)+(minint((*m)-(*p), (*m)-(*q))-r)+i-1)-toCmplx(math.Cos(theta.Get(i-1))))
	}

	//     Compute norm( U1'*X11*V1 - D11 ) / ( maxint(1,P,Q)*EPS2 ) .
	resid = golapack.Zlange('1', p, q, xf, ldx, rwork)
	result.Set(0, (resid/float64(maxint(1, *p, *q)))/eps2)

	//     Compute norm( U1'*X12*V2 - D12 ) / ( maxint(1,P,M-Q)*EPS2 ) .
	resid = golapack.Zlange('1', p, toPtr((*m)-(*q)), xf.Off(0, (*q)+1-1), ldx, rwork)
	result.Set(1, (resid/float64(maxint(1, *p, (*m)-(*q))))/eps2)

	//     Compute norm( U2'*X21*V1 - D21 ) / ( maxint(1,M-P,Q)*EPS2 ) .
	resid = golapack.Zlange('1', toPtr((*m)-(*p)), q, xf.Off((*p)+1-1, 0), ldx, rwork)
	result.Set(2, (resid/float64(maxint(1, (*m)-(*p), *q)))/eps2)

	//     Compute norm( U2'*X22*V2 - D22 ) / ( maxint(1,M-P,M-Q)*EPS2 ) .
	resid = golapack.Zlange('1', toPtr((*m)-(*p)), toPtr((*m)-(*q)), xf.Off((*p)+1-1, (*q)+1-1), ldx, rwork)
	result.Set(3, (resid/float64(maxint(1, (*m)-(*p), (*m)-(*q))))/eps2)

	//     Compute I - U1'*U1
	golapack.Zlaset('F', p, p, &zero, &one, work.CMatrix(*ldu1, opts), ldu1)
	err = goblas.Zherk(Upper, ConjTrans, *p, *p, -realone, u1, *ldu1, realone, work.CMatrix(*ldu1, opts), *ldu1)

	//     Compute norm( I - U'*U ) / ( maxint(1,P) * ULP ) .
	resid = golapack.Zlanhe('1', 'U', p, work.CMatrix(*ldu1, opts), ldu1, rwork)
	result.Set(4, (resid/float64(maxint(1, *p)))/ulp)

	//     Compute I - U2'*U2
	golapack.Zlaset('F', toPtr((*m)-(*p)), toPtr((*m)-(*p)), &zero, &one, work.CMatrix(*ldu2, opts), ldu2)
	err = goblas.Zherk(Upper, ConjTrans, (*m)-(*p), (*m)-(*p), -realone, u2, *ldu2, realone, work.CMatrix(*ldu2, opts), *ldu2)

	//     Compute norm( I - U2'*U2 ) / ( maxint(1,M-P) * ULP ) .
	resid = golapack.Zlanhe('1', 'U', toPtr((*m)-(*p)), work.CMatrix(*ldu2, opts), ldu2, rwork)
	result.Set(5, (resid/float64(maxint(1, (*m)-(*p))))/ulp)

	//     Compute I - V1T*V1T'
	golapack.Zlaset('F', q, q, &zero, &one, work.CMatrix(*ldv1t, opts), ldv1t)
	err = goblas.Zherk(Upper, NoTrans, *q, *q, -realone, v1t, *ldv1t, realone, work.CMatrix(*ldv1t, opts), *ldv1t)

	//     Compute norm( I - V1T*V1T' ) / ( maxint(1,Q) * ULP ) .
	resid = golapack.Zlanhe('1', 'U', q, work.CMatrix(*ldv1t, opts), ldv1t, rwork)
	result.Set(6, (resid/float64(maxint(1, *q)))/ulp)

	//     Compute I - V2T*V2T'
	golapack.Zlaset('F', toPtr((*m)-(*q)), toPtr((*m)-(*q)), &zero, &one, work.CMatrix(*ldv2t, opts), ldv2t)
	err = goblas.Zherk(Upper, NoTrans, (*m)-(*q), (*m)-(*q), -realone, v2t, *ldv2t, realone, work.CMatrix(*ldv2t, opts), *ldv2t)

	//     Compute norm( I - V2T*V2T' ) / ( maxint(1,M-Q) * ULP ) .
	resid = golapack.Zlanhe('1', 'U', toPtr((*m)-(*q)), work.CMatrix(*ldv2t, opts), ldv2t, rwork)
	result.Set(7, (resid/float64(maxint(1, (*m)-(*q))))/ulp)

	//     Check sorting
	result.Set(8, realzero)
	for i = 1; i <= r; i++ {
		if theta.Get(i-1) < realzero || theta.Get(i-1) > piover2 {
			result.Set(8, ulpinv)
		}
		if i > 1 {
			if theta.Get(i-1) < theta.Get(i-1-1) {
				result.Set(8, ulpinv)
			}
		}
	}

	//     The second half of the routine checks the 2-by-1 CSD
	golapack.Zlaset('F', q, q, &zero, &one, work.CMatrix(*ldx, opts), ldx)
	err = goblas.Zherk(Upper, ConjTrans, *q, *m, -realone, x, *ldx, realone, work.CMatrix(*ldx, opts), *ldx)
	if (*m) > 0 {
		eps2 = maxf64(ulp, golapack.Zlange('1', q, q, work.CMatrix(*ldx, opts), ldx, rwork)/float64(*m))
	} else {
		eps2 = ulp
	}
	r = minint(*p, (*m)-(*p), *q, (*m)-(*q))

	//     Copy the matrix X to the array XF.
	golapack.Zlacpy('F', m, m, x, ldx, xf, ldx)

	//     Compute the CSD
	golapack.Zuncsd2by1('Y', 'Y', 'Y', m, p, q, xf.Off(0, 0), ldx, xf.Off((*p)+1-1, 0), ldx, theta, u1, ldu1, u2, ldu2, v1t, ldv1t, work, lwork, rwork, toPtr(17*(r+2)), iwork, &info)

	//     Compute [X11;X21] := diag(U1,U2)'*[X11;X21]*V1 - [D11;D21]
	err = goblas.Zgemm(NoTrans, ConjTrans, *p, *q, *q, one, x, *ldx, v1t, *ldv1t, zero, work.CMatrix(*ldx, opts), *ldx)

	err = goblas.Zgemm(ConjTrans, NoTrans, *p, *q, *p, one, u1, *ldu1, work.CMatrix(*ldx, opts), *ldx, zero, x, *ldx)

	for i = 1; i <= minint(*p, *q)-r; i++ {
		x.Set(i-1, i-1, x.Get(i-1, i-1)-one)
	}
	for i = 1; i <= r; i++ {
		x.Set(minint(*p, *q)-r+i-1, minint(*p, *q)-r+i-1, x.Get(minint(*p, *q)-r+i-1, minint(*p, *q)-r+i-1)-toCmplx(math.Cos(theta.Get(i-1))))
	}

	err = goblas.Zgemm(NoTrans, ConjTrans, (*m)-(*p), *q, *q, one, x.Off((*p)+1-1, 0), *ldx, v1t, *ldv1t, zero, work.CMatrix(*ldx, opts), *ldx)

	err = goblas.Zgemm(ConjTrans, NoTrans, (*m)-(*p), *q, (*m)-(*p), one, u2, *ldu2, work.CMatrix(*ldx, opts), *ldx, zero, x.Off((*p)+1-1, 0), *ldx)

	for i = 1; i <= minint((*m)-(*p), *q)-r; i++ {
		x.Set((*m)-i+1-1, (*q)-i+1-1, x.Get((*m)-i+1-1, (*q)-i+1-1)-one)
	}
	for i = 1; i <= r; i++ {
		x.Set((*m)-(minint((*m)-(*p), *q)-r)+1-i-1, (*q)-(minint((*m)-(*p), *q)-r)+1-i-1, x.Get((*m)-(minint((*m)-(*p), *q)-r)+1-i-1, (*q)-(minint((*m)-(*p), *q)-r)+1-i-1)-toCmplx(math.Sin(theta.Get(r-i+1-1))))
	}

	//     Compute norm( U1'*X11*V1 - D11 ) / ( maxint(1,P,Q)*EPS2 ) .
	resid = golapack.Zlange('1', p, q, x, ldx, rwork)
	result.Set(9, (resid/float64(maxint(1, *p, *q)))/eps2)

	//     Compute norm( U2'*X21*V1 - D21 ) / ( maxint(1,M-P,Q)*EPS2 ) .
	resid = golapack.Zlange('1', toPtr((*m)-(*p)), q, x.Off((*p)+1-1, 0), ldx, rwork)
	result.Set(10, (resid/float64(maxint(1, (*m)-(*p), *q)))/eps2)

	//     Compute I - U1'*U1
	golapack.Zlaset('F', p, p, &zero, &one, work.CMatrix(*ldu1, opts), ldu1)
	err = goblas.Zherk(Upper, ConjTrans, *p, *p, -realone, u1, *ldu1, realone, work.CMatrix(*ldu1, opts), *ldu1)

	//     Compute norm( I - U'*U ) / ( maxint(1,P) * ULP ) .
	resid = golapack.Zlanhe('1', 'U', p, work.CMatrix(*ldu1, opts), ldu1, rwork)
	result.Set(11, (resid/float64(maxint(1, *p)))/ulp)

	//     Compute I - U2'*U2
	golapack.Zlaset('F', toPtr((*m)-(*p)), toPtr((*m)-(*p)), &zero, &one, work.CMatrix(*ldu2, opts), ldu2)
	err = goblas.Zherk(Upper, ConjTrans, (*m)-(*p), (*m)-(*p), -realone, u2, *ldu2, realone, work.CMatrix(*ldu2, opts), *ldu2)

	//     Compute norm( I - U2'*U2 ) / ( maxint(1,M-P) * ULP ) .
	resid = golapack.Zlanhe('1', 'U', toPtr((*m)-(*p)), work.CMatrix(*ldu2, opts), ldu2, rwork)
	result.Set(12, (resid/float64(maxint(1, (*m)-(*p))))/ulp)

	//     Compute I - V1T*V1T'
	golapack.Zlaset('F', q, q, &zero, &one, work.CMatrix(*ldv1t, opts), ldv1t)
	err = goblas.Zherk(Upper, NoTrans, *q, *q, -realone, v1t, *ldv1t, realone, work.CMatrix(*ldv1t, opts), *ldv1t)

	//     Compute norm( I - V1T*V1T' ) / ( maxint(1,Q) * ULP ) .
	resid = golapack.Zlanhe('1', 'U', q, work.CMatrix(*ldv1t, opts), ldv1t, rwork)
	result.Set(13, (resid/float64(maxint(1, *q)))/ulp)

	//     Check sorting
	result.Set(14, realzero)
	for i = 1; i <= r; i++ {
		if theta.Get(i-1) < realzero || theta.Get(i-1) > piover2 {
			result.Set(14, ulpinv)
		}
		if i > 1 {
			if theta.Get(i-1) < theta.Get(i-1-1) {
				result.Set(14, ulpinv)
			}
		}
	}
}
