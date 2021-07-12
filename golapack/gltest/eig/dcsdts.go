package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dcsdts tests DORCSD, which, given an M-by-M partitioned orthogonal
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
//                           = [---------------------] = [ D11 D12 ] ,
//                             [  0  0  0 |  I  0  0 ]   [ D21 D22 ]
//                             [  0  S  0 |  0  C  0 ]
//                             [  0  0  I |  0  0  0 ]
//
// and also DORCSD2BY1, which, given
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
func Dcsdts(m, p, q *int, x, xf *mat.Matrix, ldx *int, u1 *mat.Matrix, ldu1 *int, u2 *mat.Matrix, ldu2 *int, v1t *mat.Matrix, ldv1t *int, v2t *mat.Matrix, ldv2t *int, theta *mat.Vector, iwork *[]int, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var eps2, one, piover2, realone, realzero, resid, ulp, ulpinv, zero float64
	var i, info, r int
	var err error
	_ = err

	piover2 = 1.57079632679489662
	realone = 1.0
	realzero = 0.0
	zero = 0.0
	one = 1.0

	ulp = golapack.Dlamch(Precision)
	ulpinv = realone / ulp

	//     The first half of the routine checks the 2-by-2 CSD
	golapack.Dlaset('F', m, m, &zero, &one, work.Matrix(*ldx, opts), ldx)
	err = goblas.Dsyrk(Upper, ConjTrans, *m, *m, -one, x, one, work.Matrix(*ldx, opts))
	if (*m) > 0 {
		eps2 = math.Max(ulp, golapack.Dlange('1', m, m, work.Matrix(*ldx, opts), ldx, rwork)/float64(*m))
	} else {
		eps2 = ulp
	}
	r = min(*p, (*m)-(*p), *q, (*m)-(*q))

	//     Copy the matrix X to the array XF.
	golapack.Dlacpy('F', m, m, x, ldx, xf, ldx)

	//     Compute the CSD
	golapack.Dorcsd('Y', 'Y', 'Y', 'Y', 'N', 'D', m, p, q, xf, ldx, xf.Off(0, (*q)), ldx, xf.Off((*p), 0), ldx, xf.Off((*p), (*q)), ldx, theta, u1, ldu1, u2, ldu2, v1t, ldv1t, v2t, ldv2t, work, lwork, iwork, &info)

	//     Compute XF := diag(U1,U2)'*X*diag(V1,V2) - [D11 D12; D21 D22]
	golapack.Dlacpy('F', m, m, x, ldx, xf, ldx)

	err = goblas.Dgemm(NoTrans, ConjTrans, *p, *q, *q, one, xf, v1t, zero, work.Matrix(*ldx, opts))

	err = goblas.Dgemm(ConjTrans, NoTrans, *p, *q, *p, one, u1, work.Matrix(*ldx, opts), zero, xf)

	for i = 1; i <= min(*p, *q)-r; i++ {
		xf.Set(i-1, i-1, xf.Get(i-1, i-1)-one)
	}
	for i = 1; i <= r; i++ {
		xf.Set(min(*p, *q)-r+i-1, min(*p, *q)-r+i-1, xf.Get(min(*p, *q)-r+i-1, min(*p, *q)-r+i-1)-math.Cos(theta.Get(i-1)))
	}

	err = goblas.Dgemm(NoTrans, ConjTrans, *p, (*m)-(*q), (*m)-(*q), one, xf.Off(0, (*q)), v2t, zero, work.Matrix(*ldx, opts))

	err = goblas.Dgemm(ConjTrans, NoTrans, *p, (*m)-(*q), *p, one, u1, work.Matrix(*ldx, opts), zero, xf.Off(0, (*q)))

	for i = 1; i <= min(*p, (*m)-(*q))-r; i++ {
		xf.Set((*p)-i, (*m)-i, xf.Get((*p)-i, (*m)-i)+one)
	}
	for i = 1; i <= r; i++ {
		xf.Set((*p)-(min(*p, (*m)-(*q))-r)+1-i-1, (*m)-(min(*p, (*m)-(*q))-r)+1-i-1, xf.Get((*p)-(min(*p, (*m)-(*q))-r)+1-i-1, (*m)-(min(*p, (*m)-(*q))-r)+1-i-1)+math.Sin(theta.Get(r-i)))
	}

	err = goblas.Dgemm(NoTrans, ConjTrans, (*m)-(*p), *q, *q, one, xf.Off((*p), 0), v1t, zero, work.Matrix(*ldx, opts))

	err = goblas.Dgemm(ConjTrans, NoTrans, (*m)-(*p), *q, (*m)-(*p), one, u2, work.Matrix(*ldx, opts), zero, xf.Off((*p), 0))

	for i = 1; i <= min((*m)-(*p), *q)-r; i++ {
		xf.Set((*m)-i, (*q)-i, xf.Get((*m)-i, (*q)-i)-one)
	}
	for i = 1; i <= r; i++ {
		xf.Set((*m)-(min((*m)-(*p), *q)-r)+1-i-1, (*q)-(min((*m)-(*p), *q)-r)+1-i-1, xf.Get((*m)-(min((*m)-(*p), *q)-r)+1-i-1, (*q)-(min((*m)-(*p), *q)-r)+1-i-1)-math.Sin(theta.Get(r-i)))
	}

	err = goblas.Dgemm(NoTrans, ConjTrans, (*m)-(*p), (*m)-(*q), (*m)-(*q), one, xf.Off((*p), (*q)), v2t, zero, work.Matrix(*ldx, opts))

	err = goblas.Dgemm(ConjTrans, NoTrans, (*m)-(*p), (*m)-(*q), (*m)-(*p), one, u2, work.Matrix(*ldx, opts), zero, xf.Off((*p), (*q)))

	for i = 1; i <= min((*m)-(*p), (*m)-(*q))-r; i++ {
		xf.Set((*p)+i-1, (*q)+i-1, xf.Get((*p)+i-1, (*q)+i-1)-one)
	}
	for i = 1; i <= r; i++ {
		xf.Set((*p)+(min((*m)-(*p), (*m)-(*q))-r)+i-1, (*q)+(min((*m)-(*p), (*m)-(*q))-r)+i-1, xf.Get((*p)+(min((*m)-(*p), (*m)-(*q))-r)+i-1, (*q)+(min((*m)-(*p), (*m)-(*q))-r)+i-1)-math.Cos(theta.Get(i-1)))
	}

	//     Compute norm( U1'*X11*V1 - D11 ) / ( max(1,P,Q)*EPS2 ) .
	resid = golapack.Dlange('1', p, q, xf, ldx, rwork)
	result.Set(0, (resid/float64(max(1, *p, *q)))/eps2)

	//     Compute norm( U1'*X12*V2 - D12 ) / ( max(1,P,M-Q)*EPS2 ) .
	resid = golapack.Dlange('1', p, toPtr((*m)-(*q)), xf.Off(0, (*q)), ldx, rwork)
	result.Set(1, (resid/float64(max(1, *p, (*m)-(*q))))/eps2)

	//     Compute norm( U2'*X21*V1 - D21 ) / ( max(1,M-P,Q)*EPS2 ) .
	resid = golapack.Dlange('1', toPtr((*m)-(*p)), q, xf.Off((*p), 0), ldx, rwork)
	result.Set(2, (resid/float64(max(1, (*m)-(*p), *q)))/eps2)

	//     Compute norm( U2'*X22*V2 - D22 ) / ( max(1,M-P,M-Q)*EPS2 ) .
	resid = golapack.Dlange('1', toPtr((*m)-(*p)), toPtr((*m)-(*q)), xf.Off((*p), (*q)), ldx, rwork)
	result.Set(3, (resid/float64(max(1, (*m)-(*p), (*m)-(*q))))/eps2)

	//     Compute I - U1'*U1
	golapack.Dlaset('F', p, p, &zero, &one, work.Matrix(*ldu1, opts), ldu1)
	err = goblas.Dsyrk(Upper, ConjTrans, *p, *p, -one, u1, one, work.Matrix(*ldu1, opts))

	//     Compute norm( I - U'*U ) / ( max(1,P) * ULP ) .
	resid = golapack.Dlansy('1', 'U', p, work.Matrix(*ldu1, opts), ldu1, rwork)
	result.Set(4, (resid/float64(max(1, *p)))/ulp)

	//     Compute I - U2'*U2
	golapack.Dlaset('F', toPtr((*m)-(*p)), toPtr((*m)-(*p)), &zero, &one, work.Matrix(*ldu2, opts), ldu2)
	err = goblas.Dsyrk(Upper, ConjTrans, (*m)-(*p), (*m)-(*p), -one, u2, one, work.Matrix(*ldu2, opts))

	//     Compute norm( I - U2'*U2 ) / ( max(1,M-P) * ULP ) .
	resid = golapack.Dlansy('1', 'U', toPtr((*m)-(*p)), work.Matrix(*ldu2, opts), ldu2, rwork)
	result.Set(5, (resid/float64(max(1, (*m)-(*p))))/ulp)

	//     Compute I - V1T*V1T'
	golapack.Dlaset('F', q, q, &zero, &one, work.Matrix(*ldv1t, opts), ldv1t)
	err = goblas.Dsyrk(Upper, NoTrans, *q, *q, -one, v1t, one, work.Matrix(*ldv1t, opts))

	//     Compute norm( I - V1T*V1T' ) / ( max(1,Q) * ULP ) .
	resid = golapack.Dlansy('1', 'U', q, work.Matrix(*ldv1t, opts), ldv1t, rwork)
	result.Set(6, (resid/float64(max(1, *q)))/ulp)

	//     Compute I - V2T*V2T'
	golapack.Dlaset('F', toPtr((*m)-(*q)), toPtr((*m)-(*q)), &zero, &one, work.Matrix(*ldv2t, opts), ldv2t)
	err = goblas.Dsyrk(Upper, NoTrans, (*m)-(*q), (*m)-(*q), -one, v2t, one, work.Matrix(*ldv2t, opts))

	//     Compute norm( I - V2T*V2T' ) / ( max(1,M-Q) * ULP ) .
	resid = golapack.Dlansy('1', 'U', toPtr((*m)-(*q)), work.Matrix(*ldv2t, opts), ldv2t, rwork)
	result.Set(7, (resid/float64(max(1, (*m)-(*q))))/ulp)

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
	golapack.Dlaset('F', q, q, &zero, &one, work.Matrix(*ldx, opts), ldx)
	err = goblas.Dsyrk(Upper, ConjTrans, *q, *m, -one, x, one, work.Matrix(*ldx, opts))
	if (*m) > 0 {
		eps2 = math.Max(ulp, golapack.Dlange('1', q, q, work.Matrix(*ldx, opts), ldx, rwork)/float64(*m))
	} else {
		eps2 = ulp
	}
	r = min(*p, (*m)-(*p), *q, (*m)-(*q))

	//     Copy the matrix [ X11; X21 ] to the array XF.
	golapack.Dlacpy('F', m, q, x, ldx, xf, ldx)

	//     Compute the CSD
	golapack.Dorcsd2by1('Y', 'Y', 'Y', m, p, q, xf, ldx, xf.Off((*p), 0), ldx, theta, u1, ldu1, u2, ldu2, v1t, ldv1t, work, lwork, iwork, &info)

	//     Compute [X11;X21] := diag(U1,U2)'*[X11;X21]*V1 - [D11;D21]
	err = goblas.Dgemm(NoTrans, ConjTrans, *p, *q, *q, one, x, v1t, zero, work.Matrix(*ldx, opts))

	err = goblas.Dgemm(ConjTrans, NoTrans, *p, *q, *p, one, u1, work.Matrix(*ldx, opts), zero, x)

	for i = 1; i <= min(*p, *q)-r; i++ {
		x.Set(i-1, i-1, x.Get(i-1, i-1)-one)
	}
	for i = 1; i <= r; i++ {
		x.Set(min(*p, *q)-r+i-1, min(*p, *q)-r+i-1, x.Get(min(*p, *q)-r+i-1, min(*p, *q)-r+i-1)-math.Cos(theta.Get(i-1)))
	}

	err = goblas.Dgemm(NoTrans, ConjTrans, (*m)-(*p), *q, *q, one, x.Off((*p), 0), v1t, zero, work.Matrix(*ldx, opts))

	err = goblas.Dgemm(ConjTrans, NoTrans, (*m)-(*p), *q, (*m)-(*p), one, u2, work.Matrix(*ldx, opts), zero, x.Off((*p), 0))

	for i = 1; i <= min((*m)-(*p), *q)-r; i++ {
		x.Set((*m)-i, (*q)-i, x.Get((*m)-i, (*q)-i)-one)
	}
	for i = 1; i <= r; i++ {
		x.Set((*m)-(min((*m)-(*p), *q)-r)+1-i-1, (*q)-(min((*m)-(*p), *q)-r)+1-i-1, x.Get((*m)-(min((*m)-(*p), *q)-r)+1-i-1, (*q)-(min((*m)-(*p), *q)-r)+1-i-1)-math.Sin(theta.Get(r-i)))
	}

	//     Compute norm( U1'*X11*V1 - D11 ) / ( max(1,P,Q)*EPS2 ) .
	resid = golapack.Dlange('1', p, q, x, ldx, rwork)
	result.Set(9, (resid/float64(max(1, *p, *q)))/eps2)

	//     Compute norm( U2'*X21*V1 - D21 ) / ( max(1,M-P,Q)*EPS2 ) .
	resid = golapack.Dlange('1', toPtr((*m)-(*p)), q, x.Off((*p), 0), ldx, rwork)
	result.Set(10, (resid/float64(max(1, (*m)-(*p), *q)))/eps2)

	//     Compute I - U1'*U1
	golapack.Dlaset('F', p, p, &zero, &one, work.Matrix(*ldu1, opts), ldu1)
	err = goblas.Dsyrk(Upper, ConjTrans, *p, *p, -one, u1, one, work.Matrix(*ldu1, opts))

	//     Compute norm( I - U1'*U1 ) / ( max(1,P) * ULP ) .
	resid = golapack.Dlansy('1', 'U', p, work.Matrix(*ldu1, opts), ldu1, rwork)
	result.Set(11, (resid/float64(max(1, *p)))/ulp)

	//     Compute I - U2'*U2
	golapack.Dlaset('F', toPtr((*m)-(*p)), toPtr((*m)-(*p)), &zero, &one, work.Matrix(*ldu2, opts), ldu2)
	err = goblas.Dsyrk(Upper, ConjTrans, (*m)-(*p), (*m)-(*p), -one, u2, one, work.Matrix(*ldu2, opts))

	//     Compute norm( I - U2'*U2 ) / ( max(1,M-P) * ULP ) .
	resid = golapack.Dlansy('1', 'U', toPtr((*m)-(*p)), work.Matrix(*ldu2, opts), ldu2, rwork)
	result.Set(12, (resid/float64(max(1, (*m)-(*p))))/ulp)

	//     Compute I - V1T*V1T'
	golapack.Dlaset('F', q, q, &zero, &one, work.Matrix(*ldv1t, opts), ldv1t)
	err = goblas.Dsyrk(Upper, NoTrans, *q, *q, -one, v1t, one, work.Matrix(*ldv1t, opts))

	//     Compute norm( I - V1T*V1T' ) / ( max(1,Q) * ULP ) .
	resid = golapack.Dlansy('1', 'U', q, work.Matrix(*ldv1t, opts), ldv1t, rwork)
	result.Set(13, (resid/float64(max(1, *q)))/ulp)

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
