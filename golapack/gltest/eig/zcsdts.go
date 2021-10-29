package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zcsdts tests Zuncsd, which, given an M-by-M partitioned unitary
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
func zcsdts(m, p, q int, x, xf, u1, u2, v1t, v2t *mat.CMatrix, theta *mat.Vector, iwork []int, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
	var one, zero complex128
	var eps2, piover2, realone, realzero, resid, ulp, ulpinv float64
	var i, r int
	var err error

	piover2 = 1.57079632679489662
	realone = 1.0
	realzero = 0.0
	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	ulp = golapack.Dlamch(Precision)
	ulpinv = realone / ulp

	//     The first half of the routine checks the 2-by-2 CSD
	golapack.Zlaset(Full, m, m, zero, one, work.CMatrix(x.Rows, opts))
	if err = goblas.Zherk(Upper, ConjTrans, m, m, -realone, x, realone, work.CMatrix(x.Rows, opts)); err != nil {
		panic(err)
	}
	if m > 0 {
		eps2 = math.Max(ulp, golapack.Zlange('1', m, m, work.CMatrix(x.Rows, opts), rwork)/float64(m))
	} else {
		eps2 = ulp
	}
	r = min(p, m-p, q, m-q)

	//     Copy the matrix X to the array XF.
	golapack.Zlacpy(Full, m, m, x, xf)

	//     Compute the CSD
	if _, err = golapack.Zuncsd('Y', 'Y', 'Y', 'Y', NoTrans, 'D', m, p, q, xf.Off(0, 0), xf.Off(0, q), xf.Off(p, 0), xf.Off(p, q), theta, u1, u2, v1t, v2t, work, lwork, rwork, 17*(r+2), &iwork); err != nil {
		panic(err)
	}

	//     Compute XF := diag(U1,U2)'*X*diag(V1,V2) - [D11 D12; D21 D22]
	golapack.Zlacpy(Full, m, m, x, xf)

	if err = goblas.Zgemm(NoTrans, ConjTrans, p, q, q, one, xf, v1t, zero, work.CMatrix(x.Rows, opts)); err != nil {
		panic(err)
	}

	if err = goblas.Zgemm(ConjTrans, NoTrans, p, q, p, one, u1, work.CMatrix(x.Rows, opts), zero, xf); err != nil {
		panic(err)
	}

	for i = 1; i <= min(p, q)-r; i++ {
		xf.Set(i-1, i-1, xf.Get(i-1, i-1)-one)
	}
	for i = 1; i <= r; i++ {
		xf.Set(min(p, q)-r+i-1, min(p, q)-r+i-1, xf.Get(min(p, q)-r+i-1, min(p, q)-r+i-1)-toCmplx(math.Cos(theta.Get(i-1))))
	}

	if err = goblas.Zgemm(NoTrans, ConjTrans, p, m-q, m-q, one, xf.Off(0, q), v2t, zero, work.CMatrix(x.Rows, opts)); err != nil {
		panic(err)
	}

	if err = goblas.Zgemm(ConjTrans, NoTrans, p, m-q, p, one, u1, work.CMatrix(x.Rows, opts), zero, xf.Off(0, q)); err != nil {
		panic(err)
	}

	for i = 1; i <= min(p, m-q)-r; i++ {
		xf.Set(p-i, m-i, xf.Get(p-i, m-i)+one)
	}
	for i = 1; i <= r; i++ {
		xf.Set(p-(min(p, m-q)-r)+1-i-1, m-(min(p, m-q)-r)+1-i-1, xf.Get(p-(min(p, m-q)-r)+1-i-1, m-(min(p, m-q)-r)+1-i-1)+toCmplx(math.Sin(theta.Get(r-i))))
	}

	if err = goblas.Zgemm(NoTrans, ConjTrans, m-p, q, q, one, xf.Off(p, 0), v1t, zero, work.CMatrix(x.Rows, opts)); err != nil {
		panic(err)
	}

	if err = goblas.Zgemm(ConjTrans, NoTrans, m-p, q, m-p, one, u2, work.CMatrix(x.Rows, opts), zero, xf.Off(p, 0)); err != nil {
		panic(err)
	}

	for i = 1; i <= min(m-p, q)-r; i++ {
		xf.Set(m-i, q-i, xf.Get(m-i, q-i)-one)
	}
	for i = 1; i <= r; i++ {
		xf.Set(m-(min(m-p, q)-r)+1-i-1, q-(min(m-p, q)-r)+1-i-1, xf.Get(m-(min(m-p, q)-r)+1-i-1, q-(min(m-p, q)-r)+1-i-1)-toCmplx(math.Sin(theta.Get(r-i))))
	}

	if err = goblas.Zgemm(NoTrans, ConjTrans, m-p, m-q, m-q, one, xf.Off(p, q), v2t, zero, work.CMatrix(x.Rows, opts)); err != nil {
		panic(err)
	}

	if err = goblas.Zgemm(ConjTrans, NoTrans, m-p, m-q, m-p, one, u2, work.CMatrix(x.Rows, opts), zero, xf.Off(p, q)); err != nil {
		panic(err)
	}

	for i = 1; i <= min(m-p, m-q)-r; i++ {
		xf.Set(p+i-1, q+i-1, xf.Get(p+i-1, q+i-1)-one)
	}
	for i = 1; i <= r; i++ {
		xf.Set(p+(min(m-p, m-q)-r)+i-1, q+(min(m-p, m-q)-r)+i-1, xf.Get(p+(min(m-p, m-q)-r)+i-1, q+(min(m-p, m-q)-r)+i-1)-toCmplx(math.Cos(theta.Get(i-1))))
	}

	//     Compute norm( U1'*X11*V1 - D11 ) / ( max(1,P,Q)*EPS2 ) .
	resid = golapack.Zlange('1', p, q, xf, rwork)
	result.Set(0, (resid/float64(max(1, p, q)))/eps2)

	//     Compute norm( U1'*X12*V2 - D12 ) / ( max(1,P,M-Q)*EPS2 ) .
	resid = golapack.Zlange('1', p, m-q, xf.Off(0, q), rwork)
	result.Set(1, (resid/float64(max(1, p, m-q)))/eps2)

	//     Compute norm( U2'*X21*V1 - D21 ) / ( max(1,M-P,Q)*EPS2 ) .
	resid = golapack.Zlange('1', m-p, q, xf.Off(p, 0), rwork)
	result.Set(2, (resid/float64(max(1, m-p, q)))/eps2)

	//     Compute norm( U2'*X22*V2 - D22 ) / ( max(1,M-P,M-Q)*EPS2 ) .
	resid = golapack.Zlange('1', m-p, m-q, xf.Off(p, q), rwork)
	result.Set(3, (resid/float64(max(1, m-p, m-q)))/eps2)

	//     Compute I - U1'*U1
	golapack.Zlaset(Full, p, p, zero, one, work.CMatrix(u1.Rows, opts))
	if err = goblas.Zherk(Upper, ConjTrans, p, p, -realone, u1, realone, work.CMatrix(u1.Rows, opts)); err != nil {
		panic(err)
	}

	//     Compute norm( I - U'*U ) / ( max(1,P) * ULP ) .
	resid = golapack.Zlanhe('1', Upper, p, work.CMatrix(u1.Rows, opts), rwork)
	result.Set(4, (resid/float64(max(1, p)))/ulp)

	//     Compute I - U2'*U2
	golapack.Zlaset(Full, m-p, m-p, zero, one, work.CMatrix(u2.Rows, opts))
	if err = goblas.Zherk(Upper, ConjTrans, m-p, m-p, -realone, u2, realone, work.CMatrix(u2.Rows, opts)); err != nil {
		panic(err)
	}

	//     Compute norm( I - U2'*U2 ) / ( max(1,M-P) * ULP ) .
	resid = golapack.Zlanhe('1', Upper, m-p, work.CMatrix(u2.Rows, opts), rwork)
	result.Set(5, (resid/float64(max(1, m-p)))/ulp)

	//     Compute I - V1T*V1T'
	golapack.Zlaset(Full, q, q, zero, one, work.CMatrix(v1t.Rows, opts))
	if err = goblas.Zherk(Upper, NoTrans, q, q, -realone, v1t, realone, work.CMatrix(v1t.Rows, opts)); err != nil {
		panic(err)
	}

	//     Compute norm( I - V1T*V1T' ) / ( max(1,Q) * ULP ) .
	resid = golapack.Zlanhe('1', Upper, q, work.CMatrix(v1t.Rows, opts), rwork)
	result.Set(6, (resid/float64(max(1, q)))/ulp)

	//     Compute I - V2T*V2T'
	golapack.Zlaset(Full, m-q, m-q, zero, one, work.CMatrix(v2t.Rows, opts))
	if err = goblas.Zherk(Upper, NoTrans, m-q, m-q, -realone, v2t, realone, work.CMatrix(v2t.Rows, opts)); err != nil {
		panic(err)
	}

	//     Compute norm( I - V2T*V2T' ) / ( max(1,M-Q) * ULP ) .
	resid = golapack.Zlanhe('1', Upper, m-q, work.CMatrix(v2t.Rows, opts), rwork)
	result.Set(7, (resid/float64(max(1, m-q)))/ulp)

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
	golapack.Zlaset(Full, q, q, zero, one, work.CMatrix(x.Rows, opts))
	if err = goblas.Zherk(Upper, ConjTrans, q, m, -realone, x, realone, work.CMatrix(x.Rows, opts)); err != nil {
		panic(err)
	}
	if m > 0 {
		eps2 = math.Max(ulp, golapack.Zlange('1', q, q, work.CMatrix(x.Rows, opts), rwork)/float64(m))
	} else {
		eps2 = ulp
	}
	r = min(p, m-p, q, m-q)

	//     Copy the matrix X to the array XF.
	golapack.Zlacpy(Full, m, m, x, xf)

	//     Compute the CSD
	if _, err = golapack.Zuncsd2by1('Y', 'Y', 'Y', m, p, q, xf.Off(0, 0), xf.Off(p, 0), theta, u1, u2, v1t, work, lwork, rwork, 17*(r+2), &iwork); err != nil {
		panic(err)
	}

	//     Compute [X11;X21] := diag(U1,U2)'*[X11;X21]*V1 - [D11;D21]
	if err = goblas.Zgemm(NoTrans, ConjTrans, p, q, q, one, x, v1t, zero, work.CMatrix(x.Rows, opts)); err != nil {
		panic(err)
	}

	if err = goblas.Zgemm(ConjTrans, NoTrans, p, q, p, one, u1, work.CMatrix(x.Rows, opts), zero, x); err != nil {
		panic(err)
	}

	for i = 1; i <= min(p, q)-r; i++ {
		x.Set(i-1, i-1, x.Get(i-1, i-1)-one)
	}
	for i = 1; i <= r; i++ {
		x.Set(min(p, q)-r+i-1, min(p, q)-r+i-1, x.Get(min(p, q)-r+i-1, min(p, q)-r+i-1)-toCmplx(math.Cos(theta.Get(i-1))))
	}

	if err = goblas.Zgemm(NoTrans, ConjTrans, m-p, q, q, one, x.Off(p, 0), v1t, zero, work.CMatrix(x.Rows, opts)); err != nil {
		panic(err)
	}

	if err = goblas.Zgemm(ConjTrans, NoTrans, m-p, q, m-p, one, u2, work.CMatrix(x.Rows, opts), zero, x.Off(p, 0)); err != nil {
		panic(err)
	}

	for i = 1; i <= min(m-p, q)-r; i++ {
		x.Set(m-i, q-i, x.Get(m-i, q-i)-one)
	}
	for i = 1; i <= r; i++ {
		x.Set(m-(min(m-p, q)-r)+1-i-1, q-(min(m-p, q)-r)+1-i-1, x.Get(m-(min(m-p, q)-r)+1-i-1, q-(min(m-p, q)-r)+1-i-1)-toCmplx(math.Sin(theta.Get(r-i))))
	}

	//     Compute norm( U1'*X11*V1 - D11 ) / ( max(1,P,Q)*EPS2 ) .
	resid = golapack.Zlange('1', p, q, x, rwork)
	result.Set(9, (resid/float64(max(1, p, q)))/eps2)

	//     Compute norm( U2'*X21*V1 - D21 ) / ( max(1,M-P,Q)*EPS2 ) .
	resid = golapack.Zlange('1', m-p, q, x.Off(p, 0), rwork)
	result.Set(10, (resid/float64(max(1, m-p, q)))/eps2)

	//     Compute I - U1'*U1
	golapack.Zlaset(Full, p, p, zero, one, work.CMatrix(u1.Rows, opts))
	if err = goblas.Zherk(Upper, ConjTrans, p, p, -realone, u1, realone, work.CMatrix(u1.Rows, opts)); err != nil {
		panic(err)
	}

	//     Compute norm( I - U'*U ) / ( max(1,P) * ULP ) .
	resid = golapack.Zlanhe('1', Upper, p, work.CMatrix(u1.Rows, opts), rwork)
	result.Set(11, (resid/float64(max(1, p)))/ulp)

	//     Compute I - U2'*U2
	golapack.Zlaset(Full, m-p, m-p, zero, one, work.CMatrix(u2.Rows, opts))
	if err = goblas.Zherk(Upper, ConjTrans, m-p, m-p, -realone, u2, realone, work.CMatrix(u2.Rows, opts)); err != nil {
		panic(err)
	}

	//     Compute norm( I - U2'*U2 ) / ( max(1,M-P) * ULP ) .
	resid = golapack.Zlanhe('1', Upper, m-p, work.CMatrix(u2.Rows, opts), rwork)
	result.Set(12, (resid/float64(max(1, m-p)))/ulp)

	//     Compute I - V1T*V1T'
	golapack.Zlaset(Full, q, q, zero, one, work.CMatrix(v1t.Rows, opts))
	if err = goblas.Zherk(Upper, NoTrans, q, q, -realone, v1t, realone, work.CMatrix(v1t.Rows, opts)); err != nil {
		panic(err)
	}

	//     Compute norm( I - V1T*V1T' ) / ( max(1,Q) * ULP ) .
	resid = golapack.Zlanhe('1', Upper, q, work.CMatrix(v1t.Rows, opts), rwork)
	result.Set(13, (resid/float64(max(1, q)))/ulp)

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
