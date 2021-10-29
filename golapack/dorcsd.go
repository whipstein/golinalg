package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorcsd computes the CS decomposition of an M-by-M partitioned
// orthogonal matrix X:
//
//                                 [  I  0  0 |  0  0  0 ]
//                                 [  0  C  0 |  0 -S  0 ]
//     [ X11 | X12 ]   [ U1 |    ] [  0  0  0 |  0  0 -I ] [ V1 |    ]**T
// X = [-----------] = [---------] [---------------------] [---------]   .
//     [ X21 | X22 ]   [    | U2 ] [  0  0  0 |  I  0  0 ] [    | V2 ]
//                                 [  0  S  0 |  0  C  0 ]
//                                 [  0  0  I |  0  0  0 ]
//
// X11 is P-by-Q. The orthogonal matrices U1, U2, V1, and V2 are P-by-P,
// (M-P)-by-(M-P), Q-by-Q, and (M-Q)-by-(M-Q), respectively. C and S are
// R-by-R nonnegative diagonal matrices satisfying C^2 + S^2 = I, in
// which R = MIN(P,M-P,Q,M-Q).
func Dorcsd(jobu1, jobu2, jobv1t, jobv2t byte, trans mat.MatTrans, signs byte, m, p, q int, x11, x12, x21, x22 *mat.Matrix, theta *mat.Vector, u1, u2, v1t, v2t *mat.Matrix, work *mat.Vector, lwork int, iwork *[]int) (info int, err error) {
	var colmajor, defaultsigns, lquery, wantu1, wantu2, wantv1t, wantv2t bool
	var signst byte
	var transt mat.MatTrans
	var one, zero float64
	var i, ib11d, ib11e, ib12d, ib12e, ib21d, ib21e, ib22d, ib22e, ibbcsd, iorbdb, iorglq, iorgqr, iphi, itaup1, itaup2, itauq1, itauq2, j, lbbcsdwork, lbbcsdworkmin, lbbcsdworkopt, lorbdbwork, lorbdbworkopt, lorglqwork, lorglqworkmin, lorglqworkopt, lorgqrwork, lorgqrworkmin, lorgqrworkopt, lworkmin, lworkopt int

	one = 1.0
	zero = 0.0

	//     Test input arguments
	wantu1 = jobu1 == 'Y'
	wantu2 = jobu2 == 'Y'
	wantv1t = jobv1t == 'Y'
	wantv2t = jobv2t == 'Y'
	colmajor = trans != Trans
	defaultsigns = signs != 'O'
	lquery = lwork == -1
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if p < 0 || p > m {
		err = fmt.Errorf("p < 0 || p > m: p=%v, m=%v", p, m)
	} else if q < 0 || q > m {
		err = fmt.Errorf("q < 0 || q > m: q=%v, m=%v", q, m)
	} else if colmajor && x11.Rows < max(1, p) {
		err = fmt.Errorf("colmajor && x11.Rows < max(1, p): trans=%s, x11.Rows=%v, p=%v", trans, x11.Rows, p)
	} else if !colmajor && x11.Rows < max(1, q) {
		err = fmt.Errorf("!colmajor && x11.Rows < max(1, q): trans=%s, x11.Rows=%v, q=%v", trans, x11.Rows, q)
	} else if colmajor && x12.Rows < max(1, p) {
		err = fmt.Errorf("colmajor && x12.Rows < max(1, p): trans=%s, x11.Rows=%v, p=%v", trans, x11.Rows, p)
	} else if !colmajor && x12.Rows < max(1, m-q) {
		err = fmt.Errorf("!colmajor && x12.Rows < max(1, m-q): trans=%s, x11.Rows=%v, m=%v, q=%v", trans, x11.Rows, m, q)
	} else if colmajor && x21.Rows < max(1, m-p) {
		err = fmt.Errorf("colmajor && x21.Rows < max(1, m-p): trans=%s, x11.Rows=%v, m=%v, p=%v", trans, x11.Rows, m, p)
	} else if !colmajor && x21.Rows < max(1, q) {
		err = fmt.Errorf("!colmajor && x21.Rows < max(1, q): trans=%s, x11.Rows=%v, q=%v", trans, x11.Rows, q)
	} else if colmajor && x22.Rows < max(1, m-p) {
		err = fmt.Errorf("colmajor && x22.Rows < max(1, m-p): trans=%s, x11.Rows=%v, m=%v, p=%v", trans, x11.Rows, m, p)
	} else if !colmajor && x22.Rows < max(1, m-q) {
		err = fmt.Errorf("!colmajor && x22.Rows < max(1, m-q): trans=%s, x11.Rows=%v, m=%v, q=%v", trans, x11.Rows, m, q)
	} else if wantu1 && u1.Rows < p {
		err = fmt.Errorf("wantu1 && u1.Rows < p: jobu1='%c', u1.Rows=%v, p=%v", jobu1, u1.Rows, p)
	} else if wantu2 && u2.Rows < m-p {
		err = fmt.Errorf("wantu2 && u2.Rows < m-p: jobu2='%c', u2.Rows=%v, m=%v, p=%v", jobu2, u2.Rows, m, p)
	} else if wantv1t && v1t.Rows < q {
		err = fmt.Errorf("wantv1t && v1t.Rows < q: jobv1t='%c', v1t.Rows=%v, q=%v", jobv1t, v1t.Rows, q)
	} else if wantv2t && v2t.Rows < m-q {
		err = fmt.Errorf("wantv2t && v2t.Rows < m-q: jobv2t='%c', v2t.Rows=%v, m=%v, q=%v", jobv2t, v2t.Rows, m, q)
	}

	//     Work with transpose if convenient
	if err == nil && min(p, m-p) < min(q, m-q) {
		if colmajor {
			transt = Trans
		} else {
			transt = NoTrans
		}
		if defaultsigns {
			signst = 'O'
		} else {
			signst = 'D'
		}
		if info, err = Dorcsd(jobv1t, jobv2t, jobu1, jobu2, transt, signst, m, q, p, x11, x21, x12, x22, theta, v1t, v2t, u1, u2, work, lwork, iwork); err != nil {
			panic(err)
		}
		return
	}

	//     Work with permutation [ 0 I; I 0 ] * X * [ 0 I; I 0 ] if
	//     convenient
	if err == nil && m-q < q {
		if defaultsigns {
			signst = 'O'
		} else {
			signst = 'D'
		}
		if info, err = Dorcsd(jobu2, jobu1, jobv2t, jobv1t, trans, signst, m, m-p, m-q, x22, x21, x12, x11, theta, u2, u1, v2t, v1t, work, lwork, iwork); err != nil {
			panic(err)
		}
		return
	}

	//     Compute workspace
	if err == nil {

		iphi = 2
		itaup1 = iphi + max(1, q-1)
		itaup2 = itaup1 + max(1, p)
		itauq1 = itaup2 + max(1, m-p)
		itauq2 = itauq1 + max(1, q)
		iorgqr = itauq2 + max(1, m-q)
		if err = Dorgqr(m-q, m-q, m-q, u1.Off(0, 0).UpdateRows(max(1, m-q)), u1.VectorIdx(0), work, -1); err != nil {
			panic(err)
		}
		lorgqrworkopt = int(work.Get(0))
		lorgqrworkmin = max(1, m-q)
		iorglq = itauq2 + max(1, m-q)
		if err = Dorglq(m-q, m-q, m-q, u1.Off(0, 0).UpdateRows(max(1, m-q)), u1.VectorIdx(0), work, -1); err != nil {
			panic(err)
		}
		lorglqworkopt = int(work.Get(0))
		lorglqworkmin = max(1, m-q)
		iorbdb = itauq2 + max(1, m-q)
		if err = Dorbdb(trans, signs, m, p, q, x11, x12, x21, x22, theta, v1t.VectorIdx(0), u1.VectorIdx(0), u2.VectorIdx(0), v1t.VectorIdx(0), v2t.VectorIdx(0), work, -1); err != nil {
			panic(err)
		}
		lorbdbworkopt = int(work.Get(0))
		// lorbdbworkmin = lorbdbworkopt
		ib11d = itauq2 + max(1, m-q)
		ib11e = ib11d + max(1, q)
		ib12d = ib11e + max(1, q-1)
		ib12e = ib12d + max(1, q)
		ib21d = ib12e + max(1, q-1)
		ib21e = ib21d + max(1, q)
		ib22d = ib21e + max(1, q-1)
		ib22e = ib22d + max(1, q)
		ibbcsd = ib22e + max(1, q-1)
		if _, err = Dbbcsd(jobu1, jobu2, jobv1t, jobv2t, trans, m, p, q, theta, theta, u1, u2, v1t, v2t, u1.VectorIdx(0), u1.VectorIdx(0), u1.VectorIdx(0), u1.VectorIdx(0), u1.VectorIdx(0), u1.VectorIdx(0), u1.VectorIdx(0), u1.VectorIdx(0), work, -1); err != nil {
			panic(err)
		}
		lbbcsdworkopt = int(work.Get(0))
		lbbcsdworkmin = lbbcsdworkopt
		lworkopt = max(iorgqr+lorgqrworkopt, iorglq+lorglqworkopt, iorbdb+lorbdbworkopt, ibbcsd+lbbcsdworkopt) - 1
		lworkmin = max(iorgqr+lorgqrworkmin, iorglq+lorglqworkmin, iorbdb+lorbdbworkopt, ibbcsd+lbbcsdworkmin) - 1
		work.Set(0, float64(max(lworkopt, lworkmin)))

		if lwork < lworkmin && !lquery {
			err = fmt.Errorf("lwork < lworkmin && !lquery: lwork=%v, lworkmin=%v, lquery=%v", lwork, lworkmin, lquery)
		} else {
			lorgqrwork = lwork - iorgqr + 1
			lorglqwork = lwork - iorglq + 1
			lorbdbwork = lwork - iorbdb + 1
			lbbcsdwork = lwork - ibbcsd + 1
		}
	}

	//     Abort if any illegal arguments
	if err != nil {
		gltest.Xerbla2("Dorcsd", err)
		return
	} else if lquery {
		return
	}

	//     Transform to bidiagonal block form
	if err = Dorbdb(trans, signs, m, p, q, x11, x12, x21, x22, theta, work.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(itauq2-1), work.Off(iorbdb-1), lorbdbwork); err != nil {
		panic(err)
	}

	//     Accumulate Householder reflectors
	if colmajor {
		if wantu1 && p > 0 {
			Dlacpy(Lower, p, q, x11, u1)
			if err = Dorgqr(p, p, q, u1, work.Off(itaup1-1), work.Off(iorgqr-1), lorgqrwork); err != nil {
				panic(err)
			}
		}
		if wantu2 && m-p > 0 {
			Dlacpy(Lower, m-p, q, x21, u2)
			if err = Dorgqr(m-p, m-p, q, u2, work.Off(itaup2-1), work.Off(iorgqr-1), lorgqrwork); err != nil {
				panic(err)
			}
		}
		if wantv1t && q > 0 {
			Dlacpy(Upper, q-1, q-1, x11.Off(0, 1), v1t.Off(1, 1))
			v1t.Set(0, 0, one)
			for j = 2; j <= q; j++ {
				v1t.Set(0, j-1, zero)
				v1t.Set(j-1, 0, zero)
			}
			if err = Dorglq(q-1, q-1, q-1, v1t.Off(1, 1), work.Off(itauq1-1), work.Off(iorglq-1), lorglqwork); err != nil {
				panic(err)
			}
		}
		if wantv2t && m-q > 0 {
			Dlacpy(Upper, p, m-q, x12, v2t)
			if m-p > q {
				Dlacpy(Upper, m-p-q, m-p-q, x22.Off(q, p), v2t.Off(p, p))
			}
			if m > q {
				if err = Dorglq(m-q, m-q, m-q, v2t, work.Off(itauq2-1), work.Off(iorglq-1), lorglqwork); err != nil {
					panic(err)
				}
			}
		}
	} else {
		if wantu1 && p > 0 {
			Dlacpy(Upper, q, p, x11, u1)
			if err = Dorglq(p, p, q, u1, work.Off(itaup1-1), work.Off(iorglq-1), lorglqwork); err != nil {
				panic(err)
			}
		}
		if wantu2 && m-p > 0 {
			Dlacpy(Upper, q, m-p, x21, u2)
			if err = Dorglq(m-p, m-p, q, u2, work.Off(itaup2-1), work.Off(iorglq-1), lorglqwork); err != nil {
				panic(err)
			}
		}
		if wantv1t && q > 0 {
			Dlacpy(Lower, q-1, q-1, x11.Off(1, 0), v1t.Off(1, 1))
			v1t.Set(0, 0, one)
			for j = 2; j <= q; j++ {
				v1t.Set(0, j-1, zero)
				v1t.Set(j-1, 0, zero)
			}
			if err = Dorgqr(q-1, q-1, q-1, v1t.Off(1, 1), work.Off(itauq1-1), work.Off(iorgqr-1), lorgqrwork); err != nil {
				panic(err)
			}
		}
		if wantv2t && m-q > 0 {
			Dlacpy(Lower, m-q, p, x12, v2t)
			Dlacpy(Lower, m-p-q, m-p-q, x22.Off(p, q), v2t.Off(p, p))
			if err = Dorgqr(m-q, m-q, m-q, v2t, work.Off(itauq2-1), work.Off(iorgqr-1), lorgqrwork); err != nil {
				panic(err)
			}
		}
	}

	//     Compute the CSD of the matrix in bidiagonal-block form
	if info, err = Dbbcsd(jobu1, jobu2, jobv1t, jobv2t, trans, m, p, q, theta, work.Off(iphi-1), u1, u2, v1t, v2t, work.Off(ib11d-1), work.Off(ib11e-1), work.Off(ib12d-1), work.Off(ib12e-1), work.Off(ib21d-1), work.Off(ib21e-1), work.Off(ib22d-1), work.Off(ib22e-1), work.Off(ibbcsd-1), lbbcsdwork); err != nil {
		panic(err)
	}

	//     Permute rows and columns to place identity submatrices in top-
	//     left corner of (1,1)-block and/or bottom-right corner of (1,2)-
	//     block and/or bottom-right corner of (2,1)-block and/or top-left
	//     corner of (2,2)-block
	if q > 0 && wantu2 {
		for i = 1; i <= q; i++ {
			(*iwork)[i-1] = m - p - q + i
		}
		for i = q + 1; i <= m-p; i++ {
			(*iwork)[i-1] = i - q
		}
		if colmajor {
			Dlapmt(false, m-p, m-p, u2, iwork)
		} else {
			Dlapmr(false, m-p, m-p, u2, iwork)
		}
	}
	if m > 0 && wantv2t {
		for i = 1; i <= p; i++ {
			(*iwork)[i-1] = m - p - q + i
		}
		for i = p + 1; i <= m-q; i++ {
			(*iwork)[i-1] = i - p
		}
		if !colmajor {
			Dlapmt(false, m-q, m-q, v2t, iwork)
		} else {
			Dlapmr(false, m-q, m-q, v2t, iwork)
		}
	}

	return
}
