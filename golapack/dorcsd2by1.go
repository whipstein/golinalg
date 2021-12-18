package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorcsd2by1 computes the CS decomposition of an M-by-Q matrix X with
// orthonormal columns that has been partitioned into a 2-by-1 block
// structure:
//
//                                [  I1 0  0 ]
//                                [  0  C  0 ]
//          [ X11 ]   [ U1 |    ] [  0  0  0 ]
//      X = [-----] = [---------] [----------] V1**T .
//          [ X21 ]   [    | U2 ] [  0  0  0 ]
//                                [  0  S  0 ]
//                                [  0  0  I2]
//
// X11 is P-by-Q. The orthogonal matrices U1, U2, and V1 are P-by-P,
// (M-P)-by-(M-P), and Q-by-Q, respectively. C and S are R-by-R
// nonnegative diagonal matrices satisfying C^2 + S^2 = I, in which
// R = min(P,M-P,Q,M-Q). I1 is a K1-by-K1 identity matrix and I2 is a
// K2-by-K2 identity matrix, where K1 = max(Q+P-M,0), K2 = max(Q-P,0).
func Dorcsd2by1(jobu1, jobu2, jobv1t byte, m, p, q int, x11, x21 *mat.Matrix, theta *mat.Vector, u1, u2, v1t *mat.Matrix, work *mat.Vector, lwork int, iwork *[]int) (info int, err error) {
	var lquery, wantu1, wantu2, wantv1t bool
	var one, zero float64
	var i, ib11d, ib11e, ib12d, ib12e, ib21d, ib21e, ib22d, ib22e, ibbcsd, iorbdb, iorglq, iorgqr, iphi, itaup1, itaup2, itauq1, j, lbbcsd, lorbdb, lorglq, lorglqmin, lorglqopt, lorgqr, lorgqrmin, lorgqropt, lworkmin, lworkopt, r int

	dum1 := vf(1)
	dum2 := mf(1, 1, opts)

	one = 1.0
	zero = 0.0

	//     Test input arguments
	wantu1 = jobu1 == 'Y'
	wantu2 = jobu2 == 'Y'
	wantv1t = jobv1t == 'Y'
	lquery = lwork == -1

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if p < 0 || p > m {
		err = fmt.Errorf("p < 0 || p > m: p=%v, m=%v", p, m)
	} else if q < 0 || q > m {
		err = fmt.Errorf("q < 0 || q > m: q=%v, m=%v", q, m)
	} else if x11.Rows < max(1, p) {
		err = fmt.Errorf("x11.Rows < max(1, p): x11.Rows=%v, p=%v", x11.Rows, p)
	} else if x21.Rows < max(1, m-p) {
		err = fmt.Errorf("x21.Rows < max(1, m-p): x21.Rows=%v, m=%v, p=%v", x21.Rows, m, p)
	} else if wantu1 && u1.Rows < max(1, p) {
		err = fmt.Errorf("wantu1 && u1.Rows < max(1, p): jobu1='%c', u1.Rows=%v, p=%v", jobu1, u1.Rows, p)
	} else if wantu2 && u2.Rows < max(1, m-p) {
		err = fmt.Errorf("wantu2 && u2.Rows < max(1, m-p): jobu2='%c', u2.Rows=%v, m=%v, p=%v", jobu2, u2.Rows, m, p)
	} else if wantv1t && v1t.Rows < max(1, q) {
		err = fmt.Errorf("wantv1t && v1t.Rows < max(1, q): jobv1t='%c', v1t.Rows=%v, q=%v", jobv1t, v1t.Rows, q)
	}

	r = min(p, m-p, q, m-q)

	//     Compute workspace
	//
	//       WORK layout:
	//     |-------------------------------------------------------|
	//     | LWORKOPT (1)                                          |
	//     |-------------------------------------------------------|
	//     | PHI (max(1,R-1))                                      |
	//     |-------------------------------------------------------|
	//     | TAUP1 (max(1,P))                        | B11D (R)    |
	//     | TAUP2 (max(1,M-P))                      | B11E (R-1)  |
	//     | TAUQ1 (max(1,Q))                        | B12D (R)    |
	//     |-----------------------------------------| B12E (R-1)  |
	//     | DORBDB WORK | DORGQR WORK | DORGLQ WORK | B21D (R)    |
	//     |             |             |             | B21E (R-1)  |
	//     |             |             |             | B22D (R)    |
	//     |             |             |             | B22E (R-1)  |
	//     |             |             |             | DBBCSD WORK |
	//     |-------------------------------------------------------|
	if err == nil {
		iphi = 2
		ib11d = iphi + max(1, r-1)
		ib11e = ib11d + max(1, r)
		ib12d = ib11e + max(1, r-1)
		ib12e = ib12d + max(1, r)
		ib21d = ib12e + max(1, r-1)
		ib21e = ib21d + max(1, r)
		ib22d = ib21e + max(1, r-1)
		ib22e = ib22d + max(1, r)
		ibbcsd = ib22e + max(1, r-1)
		itaup1 = iphi + max(1, r-1)
		itaup2 = itaup1 + max(1, p)
		itauq1 = itaup2 + max(1, m-p)
		iorbdb = itauq1 + max(1, q)
		iorgqr = itauq1 + max(1, q)
		iorglq = itauq1 + max(1, q)
		lorgqrmin = 1
		lorgqropt = 1
		lorglqmin = 1
		lorglqopt = 1
		if r == q {
			if err = Dorbdb1(m, p, q, x11, x21, theta, dum1, dum1, dum1, dum1, work, -1); err != nil {
				panic(err)
			}
			lorbdb = int(work.Get(0))
			if wantu1 && p > 0 {
				if err = Dorgqr(p, p, q, u1, dum1, work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, p)
				lorgqropt = max(lorgqropt, int(work.Get(0)))
			}
			if wantu2 && m-p > 0 {
				if err = Dorgqr(m-p, m-p, q, u2, dum1, work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, m-p)
				lorgqropt = max(lorgqropt, int(work.Get(0)))
			}
			if wantv1t && q > 0 {
				if err = Dorglq(q-1, q-1, q-1, v1t, dum1, work, -1); err != nil {
					panic(err)
				}
				lorglqmin = max(lorglqmin, q-1)
				lorglqopt = max(lorglqopt, int(work.Get(0)))
			}
			if _, err = Dbbcsd(jobu1, jobu2, jobv1t, 'N', NoTrans, m, p, q, theta, dum1, u1, u2, v1t, dum2, dum1, dum1, dum1, dum1, dum1, dum1, dum1, dum1, work, -1); err != nil {
				panic(err)
			}
			lbbcsd = int(work.Get(0))
		} else if r == p {
			if err = Dorbdb2(m, p, q, x11, x21, theta, dum1, dum1, dum1, dum1, work, -1); err != nil {
				panic(err)
			}
			lorbdb = int(work.Get(0))
			if wantu1 && p > 0 {
				if err = Dorgqr(p-1, p-1, p-1, u1.Off(1, 1), dum1, work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, p-1)
				lorgqropt = max(lorgqropt, int(work.Get(0)))
			}
			if wantu2 && m-p > 0 {
				if err = Dorgqr(m-p, m-p, q, u2, dum1, work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, m-p)
				lorgqropt = max(lorgqropt, int(work.Get(0)))
			}
			if wantv1t && q > 0 {
				if err = Dorglq(q, q, r, v1t, dum1, work, -1); err != nil {
					panic(err)
				}
				lorglqmin = max(lorglqmin, q)
				lorglqopt = max(lorglqopt, int(work.Get(0)))
			}
			if _, err = Dbbcsd(jobv1t, 'N', jobu1, jobu2, Trans, m, q, p, theta, dum1, v1t, dum2, u1, u2, dum1, dum1, dum1, dum1, dum1, dum1, dum1, dum1, work, -1); err != nil {
				panic(err)
			}
			lbbcsd = int(work.Get(0))
		} else if r == m-p {
			if err = Dorbdb3(m, p, q, x11, x21, theta, dum1, dum1, dum1, dum1, work, -1); err != nil {
				panic(err)
			}
			lorbdb = int(work.Get(0))
			if wantu1 && p > 0 {
				if err = Dorgqr(p, p, q, u1, dum1, work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, p)
				lorgqropt = max(lorgqropt, int(work.Get(0)))
			}
			if wantu2 && m-p > 0 {
				if err = Dorgqr(m-p-1, m-p-1, m-p-1, u2.Off(1, 1), dum1, work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, m-p-1)
				lorgqropt = max(lorgqropt, int(work.Get(0)))
			}
			if wantv1t && q > 0 {
				if err = Dorglq(q, q, r, v1t, dum1, work, -1); err != nil {
					panic(err)
				}
				lorglqmin = max(lorglqmin, q)
				lorglqopt = max(lorglqopt, int(work.Get(0)))
			}
			if _, err = Dbbcsd('N', jobv1t, jobu2, jobu1, Trans, m, m-q, m-p, theta, dum1, dum2, v1t, u2, u1, dum1, dum1, dum1, dum1, dum1, dum1, dum1, dum1, work, -1); err != nil {
				panic(err)
			}
			lbbcsd = int(work.Get(0))
		} else {
			if err = Dorbdb4(m, p, q, x11, x21, theta, dum1, dum1, dum1, dum1, dum1, work, -1); err != nil {
				panic(err)
			}
			lorbdb = m + int(work.Get(0))
			if wantu1 && p > 0 {
				if err = Dorgqr(p, p, m-q, u1, dum1, work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, p)
				lorgqropt = max(lorgqropt, int(work.Get(0)))
			}
			if wantu2 && m-p > 0 {
				if err = Dorgqr(m-p, m-p, m-q, u2, dum1, work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, m-p)
				lorgqropt = max(lorgqropt, int(work.Get(0)))
			}
			if wantv1t && q > 0 {
				if err = Dorglq(q, q, q, v1t, dum1, work, -1); err != nil {
					panic(err)
				}
				lorglqmin = max(lorglqmin, q)
				lorglqopt = max(lorglqopt, int(work.Get(0)))
			}
			if _, err = Dbbcsd(jobu2, jobu1, 'N', jobv1t, NoTrans, m, m-p, m-q, theta, dum1, u2, u1, dum2, v1t, dum1, dum1, dum1, dum1, dum1, dum1, dum1, dum1, work, -1); err != nil {
				panic(err)
			}
			lbbcsd = int(work.Get(0))
		}
		lworkmin = max(iorbdb+lorbdb-1, iorgqr+lorgqrmin-1, iorglq+lorglqmin-1, ibbcsd+lbbcsd-1)
		lworkopt = max(iorbdb+lorbdb-1, iorgqr+lorgqropt-1, iorglq+lorglqopt-1, ibbcsd+lbbcsd-1)
		work.Set(0, float64(lworkopt))
		if lwork < lworkmin && !lquery {
			err = fmt.Errorf("lwork < lworkmin && ! lquery: lwork=%v, lworkmin=%v, lquery=%v", lwork, lworkmin, lquery)
		}
	}
	if err != nil {
		gltest.Xerbla2("Dorcsd2by1", err)
		return
	} else if lquery {
		return
	}
	lorgqr = lwork - iorgqr + 1
	lorglq = lwork - iorglq + 1

	//     Handle four cases separately: R = Q, R = P, R = M-P, and R = M-Q,
	//     in which R = min(P,M-P,Q,M-Q)
	if r == q {
		//        Case 1: R = Q
		//
		//        Simultaneously bidiagonalize X11 and X21
		if err = Dorbdb1(m, p, q, x11, x21, theta, work.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(iorbdb-1), lorbdb); err != nil {
			panic(err)
		}

		//        Accumulate Householder reflectors
		if wantu1 && p > 0 {
			Dlacpy(Lower, p, q, x11, u1)
			if err = Dorgqr(p, p, q, u1, work.Off(itaup1-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantu2 && m-p > 0 {
			Dlacpy(Lower, m-p, q, x21, u2)
			if err = Dorgqr(m-p, m-p, q, u2, work.Off(itaup2-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantv1t && q > 0 {
			v1t.Set(0, 0, one)
			for j = 2; j <= q; j++ {
				v1t.Set(0, j-1, zero)
				v1t.Set(j-1, 0, zero)
			}
			Dlacpy(Upper, q-1, q-1, x21.Off(0, 1), v1t.Off(1, 1))
			if err = Dorglq(q-1, q-1, q-1, v1t.Off(1, 1), work.Off(itauq1-1), work.Off(iorglq-1), lorglq); err != nil {
				panic(err)
			}
		}

		//        Simultaneously diagonalize X11 and X21.
		if _, err = Dbbcsd(jobu1, jobu2, jobv1t, 'N', NoTrans, m, p, q, theta, work.Off(iphi-1), u1, u2, v1t, dum2, work.Off(ib11d-1), work.Off(ib11e-1), work.Off(ib12d-1), work.Off(ib12e-1), work.Off(ib21d-1), work.Off(ib21e-1), work.Off(ib22d-1), work.Off(ib22e-1), work.Off(ibbcsd-1), lbbcsd); err != nil {
			panic(err)
		}

		//        Permute rows and columns to place zero submatrices in
		//        preferred positions
		if q > 0 && wantu2 {
			for i = 1; i <= q; i++ {
				(*iwork)[i-1] = m - p - q + i
			}
			for i = q + 1; i <= m-p; i++ {
				(*iwork)[i-1] = i - q
			}
			Dlapmt(false, m-p, m-p, u2, iwork)
		}
	} else if r == p {
		//        Case 2: R = P
		//
		//        Simultaneously bidiagonalize X11 and X21
		if err = Dorbdb2(m, p, q, x11, x21, theta, work.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(iorbdb-1), lorbdb); err != nil {
			panic(err)
		}

		//        Accumulate Householder reflectors
		if wantu1 && p > 0 {
			u1.Set(0, 0, one)
			for j = 2; j <= p; j++ {
				u1.Set(0, j-1, zero)
				u1.Set(j-1, 0, zero)
			}
			Dlacpy(Lower, p-1, p-1, x11.Off(1, 0), u1.Off(1, 1))
			if err = Dorgqr(p-1, p-1, p-1, u1.Off(1, 1), work.Off(itaup1-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantu2 && m-p > 0 {
			Dlacpy(Lower, m-p, q, x21, u2)
			if err = Dorgqr(m-p, m-p, q, u2, work.Off(itaup2-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantv1t && q > 0 {
			Dlacpy(Upper, p, q, x11, v1t)
			if err = Dorglq(q, q, r, v1t, work.Off(itauq1-1), work.Off(iorglq-1), lorglq); err != nil {
				panic(err)
			}
		}

		//        Simultaneously diagonalize X11 and X21.
		if _, err = Dbbcsd(jobv1t, 'N', jobu1, jobu2, Trans, m, q, p, theta, work.Off(iphi-1), v1t, dum2, u1, u2, work.Off(ib11d-1), work.Off(ib11e-1), work.Off(ib12d-1), work.Off(ib12e-1), work.Off(ib21d-1), work.Off(ib21e-1), work.Off(ib22d-1), work.Off(ib22e-1), work.Off(ibbcsd-1), lbbcsd); err != nil {
			panic(err)
		}

		//        Permute rows and columns to place identity submatrices in
		//        preferred positions
		if q > 0 && wantu2 {
			for i = 1; i <= q; i++ {
				(*iwork)[i-1] = m - p - q + i
			}
			for i = q + 1; i <= m-p; i++ {
				(*iwork)[i-1] = i - q
			}
			Dlapmt(false, m-p, m-p, u2, iwork)
		}
	} else if r == m-p {
		//        Case 3: R = M-P
		//
		//        Simultaneously bidiagonalize X11 and X21
		if err = Dorbdb3(m, p, q, x11, x21, theta, work.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(iorbdb-1), lorbdb); err != nil {
			panic(err)
		}

		//        Accumulate Householder reflectors
		if wantu1 && p > 0 {
			Dlacpy(Lower, p, q, x11, u1)
			if err = Dorgqr(p, p, q, u1, work.Off(itaup1-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantu2 && m-p > 0 {
			u2.Set(0, 0, one)
			for j = 2; j <= m-p; j++ {
				u2.Set(0, j-1, zero)
				u2.Set(j-1, 0, zero)
			}
			Dlacpy(Lower, m-p-1, m-p-1, x21.Off(1, 0), u2.Off(1, 1))
			if err = Dorgqr(m-p-1, m-p-1, m-p-1, u2.Off(1, 1), work.Off(itaup2-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantv1t && q > 0 {
			Dlacpy(Upper, m-p, q, x21, v1t)
			if err = Dorglq(q, q, r, v1t, work.Off(itauq1-1), work.Off(iorglq-1), lorglq); err != nil {
				panic(err)
			}
		}

		//        Simultaneously diagonalize X11 and X21.
		if _, err = Dbbcsd('N', jobv1t, jobu2, jobu1, Trans, m, m-q, m-p, theta, work.Off(iphi-1), dum2, v1t, u2, u1, work.Off(ib11d-1), work.Off(ib11e-1), work.Off(ib12d-1), work.Off(ib12e-1), work.Off(ib21d-1), work.Off(ib21e-1), work.Off(ib22d-1), work.Off(ib22e-1), work.Off(ibbcsd-1), lbbcsd); err != nil {
			panic(err)
		}

		//        Permute rows and columns to place identity submatrices in
		//        preferred positions
		if q > r {
			for i = 1; i <= r; i++ {
				(*iwork)[i-1] = q - r + i
			}
			for i = r + 1; i <= q; i++ {
				(*iwork)[i-1] = i - r
			}
			if wantu1 {
				Dlapmt(false, p, q, u1, iwork)
			}
			if wantv1t {
				Dlapmr(false, q, q, v1t, iwork)
			}
		}
	} else {
		//        Case 4: R = M-Q
		//
		//        Simultaneously bidiagonalize X11 and X21
		if err = Dorbdb4(m, p, q, x11, x21, theta, work.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(iorbdb-1), work.Off(iorbdb+m-1), lorbdb-m); err != nil {
			panic(err)
		}

		//        Accumulate Householder reflectors
		if wantu1 && p > 0 {
			u1.OffIdx(0).Vector().Copy(p, work.Off(iorbdb-1), 1, 1)
			for j = 2; j <= p; j++ {
				u1.Set(0, j-1, zero)
			}
			Dlacpy(Lower, p-1, m-q-1, x11.Off(1, 0), u1.Off(1, 1))
			if err = Dorgqr(p, p, m-q, u1, work.Off(itaup1-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantu2 && m-p > 0 {
			u2.OffIdx(0).Vector().Copy(m-p, work.Off(iorbdb+p-1), 1, 1)
			for j = 2; j <= m-p; j++ {
				u2.Set(0, j-1, zero)
			}
			Dlacpy(Lower, m-p-1, m-q-1, x21.Off(1, 0), u2.Off(1, 1))
			if err = Dorgqr(m-p, m-p, m-q, u2, work.Off(itaup2-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantv1t && q > 0 {
			Dlacpy(Upper, m-q, q, x21, v1t)
			Dlacpy(Upper, p-(m-q), q-(m-q), x11.Off(m-q, m-q), v1t.Off(m-q, m-q))
			Dlacpy(Upper, -p+q, q-p, x21.Off(m-q, p), v1t.Off(p, p))
			if err = Dorglq(q, q, q, v1t, work.Off(itauq1-1), work.Off(iorglq-1), lorglq); err != nil {
				panic(err)
			}
		}

		//        Simultaneously diagonalize X11 and X21.
		if _, err = Dbbcsd(jobu2, jobu1, 'N', jobv1t, NoTrans, m, m-p, m-q, theta, work.Off(iphi-1), u2, u1, dum2, v1t, work.Off(ib11d-1), work.Off(ib11e-1), work.Off(ib12d-1), work.Off(ib12e-1), work.Off(ib21d-1), work.Off(ib21e-1), work.Off(ib22d-1), work.Off(ib22e-1), work.Off(ibbcsd-1), lbbcsd); err != nil {
			panic(err)
		}

		//        Permute rows and columns to place identity submatrices in
		//        preferred positions
		if p > r {
			for i = 1; i <= r; i++ {
				(*iwork)[i-1] = p - r + i
			}
			for i = r + 1; i <= p; i++ {
				(*iwork)[i-1] = i - r
			}
			if wantu1 {
				Dlapmt(false, p, p, u1, iwork)
			}
			if wantv1t {
				Dlapmr(false, p, q, v1t, iwork)
			}
		}
	}

	return
}
