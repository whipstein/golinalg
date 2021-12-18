package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zuncsd2by1 computes the CS decomposition of an M-by-Q matrix X with
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
// X11 is P-by-Q. The unitary matrices U1, U2, and V1 are P-by-P,
// (M-P)-by-(M-P), and Q-by-Q, respectively. C and S are R-by-R
// nonnegative diagonal matrices satisfying C^2 + S^2 = I, in which
// R = min(P,M-P,Q,M-Q). I1 is a K1-by-K1 identity matrix and I2 is a
// K2-by-K2 identity matrix, where K1 = max(Q+P-M,0), K2 = max(Q-P,0).
func Zuncsd2by1(jobu1, jobu2, jobv1t byte, m, p, q int, x11, x21 *mat.CMatrix, theta *mat.Vector, u1, u2, v1t *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, lrwork int, iwork *[]int) (info int, err error) {
	var lquery, wantu1, wantu2, wantv1t bool
	var one, zero complex128
	var i, ib11d, ib11e, ib12d, ib12e, ib21d, ib21e, ib22d, ib22e, ibbcsd, iorbdb, iorglq, iorgqr, iphi, itaup1, itaup2, itauq1, j, lbbcsd, lorbdb, lorglq, lorglqmin, lorglqopt, lorgqr, lorgqrmin, lorgqropt, lrworkmin, lrworkopt, lworkmin, lworkopt, r int

	dum := vf(1)
	cdum := cmf(1, 1, opts)

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

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
	//     |-----------------------------------------|
	//     | LWORKOPT (1)                            |
	//     |-----------------------------------------|
	//     | TAUP1 (max(1,P))                        |
	//     | TAUP2 (max(1,M-P))                      |
	//     | TAUQ1 (max(1,Q))                        |
	//     |-----------------------------------------|
	//     | ZUNBDB WORK | ZUNGQR WORK | ZUNGLQ WORK |
	//     |             |             |             |
	//     |             |             |             |
	//     |             |             |             |
	//     |             |             |             |
	//     |-----------------------------------------|
	//       RWORK layout:
	//     |------------------|
	//     | LRWORKOPT (1)    |
	//     |------------------|
	//     | PHI (max(1,R-1)) |
	//     |------------------|
	//     | B11D (R)         |
	//     | B11E (R-1)       |
	//     | B12D (R)         |
	//     | B12E (R-1)       |
	//     | B21D (R)         |
	//     | B21E (R-1)       |
	//     | B22D (R)         |
	//     | B22E (R-1)       |
	//     | ZBBCSD RWORK     |
	//     |------------------|
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
		itaup1 = 2
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
			if err = Zunbdb1(m, p, q, x11, x21, theta, dum, cdum.Off(0, 0).CVector(), cdum.Off(0, 0).CVector(), cdum.Off(0, 0).CVector(), work, -1); err != nil {
				panic(err)
			}
			lorbdb = int(work.GetRe(0))
			if wantu1 && p > 0 {
				if err = Zungqr(p, p, q, u1, cdum.Off(0, 0).CVector(), work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, p)
				lorgqropt = max(lorgqropt, int(work.GetRe(0)))
			}
			if wantu2 && m-p > 0 {
				if err = Zungqr(m-p, m-p, q, u2, cdum.Off(0, 0).CVector(), work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, m-p)
				lorgqropt = max(lorgqropt, int(work.GetRe(0)))
			}
			if wantv1t && q > 0 {
				if err = Zunglq(q-1, q-1, q-1, v1t, cdum.Off(0, 0).CVector(), work, -1); err != nil {
					panic(err)
				}
				lorglqmin = max(lorglqmin, q-1)
				lorglqopt = max(lorglqopt, int(work.GetRe(0)))
			}
			if _, err = Zbbcsd(jobu1, jobu2, jobv1t, 'N', NoTrans, m, p, q, theta, dum, u1, u2, v1t, cdum, dum, dum, dum, dum, dum, dum, dum, dum, rwork.Off(0), -1); err != nil {
				panic(err)
			}
			lbbcsd = int(rwork.Get(0))
		} else if r == p {
			if err = Zunbdb2(m, p, q, x11, x21, theta, dum, cdum.Off(0, 0).CVector(), cdum.Off(0, 0).CVector(), cdum.Off(0, 0).CVector(), work, -1); err != nil {
				panic(err)
			}
			lorbdb = int(work.GetRe(0))
			if wantu1 && p > 0 {
				if err = Zungqr(p-1, p-1, p-1, u1.Off(1, 1), cdum.Off(0, 0).CVector(), work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, p-1)
				lorgqropt = max(lorgqropt, int(work.GetRe(0)))
			}
			if wantu2 && m-p > 0 {
				if err = Zungqr(m-p, m-p, q, u2, cdum.Off(0, 0).CVector(), work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, m-p)
				lorgqropt = max(lorgqropt, int(work.GetRe(0)))
			}
			if wantv1t && q > 0 {
				if err = Zunglq(q, q, r, v1t, cdum.Off(0, 0).CVector(), work, -1); err != nil {
					panic(err)
				}
				lorglqmin = max(lorglqmin, q)
				lorglqopt = max(lorglqopt, int(work.GetRe(0)))
			}
			if _, err = Zbbcsd(jobv1t, 'N', jobu1, jobu2, Trans, m, q, p, theta, dum, v1t, cdum, u1, u2, dum, dum, dum, dum, dum, dum, dum, dum, rwork.Off(0), -1); err != nil {
				panic(err)
			}
			lbbcsd = int(rwork.Get(0))
		} else if r == m-p {
			if err = Zunbdb3(m, p, q, x11, x21, theta, dum, cdum.Off(0, 0).CVector(), cdum.Off(0, 0).CVector(), cdum.Off(0, 0).CVector(), work, -1); err != nil {
				panic(err)
			}
			lorbdb = int(work.GetRe(0))
			if wantu1 && p > 0 {
				if err = Zungqr(p, p, q, u1, cdum.Off(0, 0).CVector(), work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, p)
				lorgqropt = max(lorgqropt, int(work.GetRe(0)))
			}
			if wantu2 && m-p > 0 {
				if err = Zungqr(m-p-1, m-p-1, m-p-1, u2.Off(1, 1), cdum.Off(0, 0).CVector(), work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, m-p-1)
				lorgqropt = max(lorgqropt, int(work.GetRe(0)))
			}
			if wantv1t && q > 0 {
				if err = Zunglq(q, q, r, v1t, cdum.Off(0, 0).CVector(), work, -1); err != nil {
					panic(err)
				}
				lorglqmin = max(lorglqmin, q)
				lorglqopt = max(lorglqopt, int(work.GetRe(0)))
			}
			if _, err = Zbbcsd('N', jobv1t, jobu2, jobu1, Trans, m, m-q, m-p, theta, dum, cdum, v1t, u2, u1, dum, dum, dum, dum, dum, dum, dum, dum, rwork.Off(0), -1); err != nil {
				panic(err)
			}
			lbbcsd = int(rwork.Get(0))
		} else {
			if err = Zunbdb4(m, p, q, x11, x21, theta, dum, cdum.Off(0, 0).CVector(), cdum.Off(0, 0).CVector(), cdum.Off(0, 0).CVector(), cdum.Off(0, 0).CVector(), work, -1); err != nil {
				panic(err)
			}
			lorbdb = m + int(work.GetRe(0))
			if wantu1 && p > 0 {
				if err = Zungqr(p, p, m-q, u1, cdum.Off(0, 0).CVector(), work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, p)
				lorgqropt = max(lorgqropt, int(work.GetRe(0)))
			}
			if wantu2 && m-p > 0 {
				if err = Zungqr(m-p, m-p, m-q, u2, cdum.Off(0, 0).CVector(), work, -1); err != nil {
					panic(err)
				}
				lorgqrmin = max(lorgqrmin, m-p)
				lorgqropt = max(lorgqropt, int(work.GetRe(0)))
			}
			if wantv1t && q > 0 {
				if err = Zunglq(q, q, q, v1t, cdum.Off(0, 0).CVector(), work, -1); err != nil {
					panic(err)
				}
				lorglqmin = max(lorglqmin, q)
				lorglqopt = max(lorglqopt, int(work.GetRe(0)))
			}
			if _, err = Zbbcsd(jobu2, jobu1, 'N', jobv1t, NoTrans, m, m-p, m-q, theta, dum, u2, u1, cdum, v1t, dum, dum, dum, dum, dum, dum, dum, dum, rwork.Off(0), -1); err != nil {
				panic(err)
			}
			lbbcsd = int(rwork.Get(0))
		}
		lrworkmin = ibbcsd + lbbcsd - 1
		lrworkopt = lrworkmin
		rwork.Set(0, float64(lrworkopt))
		lworkmin = max(iorbdb+lorbdb-1, iorgqr+lorgqrmin-1, iorglq+lorglqmin-1)
		lworkopt = max(iorbdb+lorbdb-1, iorgqr+lorgqropt-1, iorglq+lorglqopt-1)
		work.SetRe(0, float64(lworkopt))
		if lwork < lworkmin && !lquery {
			err = fmt.Errorf("lwork < lworkmin && !lquery: lwork=%v, lworkmin=%v, lquery=%v", lwork, lworkmin, lquery)
		}
	}
	if err != nil {
		gltest.Xerbla2("Zuncsd2by1", err)
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
		if err = Zunbdb1(m, p, q, x11, x21, theta, rwork.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(iorbdb-1), lorbdb); err != nil {
			panic(err)
		}

		//        Accumulate Householder reflectors
		if wantu1 && p > 0 {
			Zlacpy(Lower, p, q, x11, u1)
			if err = Zungqr(p, p, q, u1, work.Off(itaup1-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantu2 && m-p > 0 {
			Zlacpy(Lower, m-p, q, x21, u2)
			if err = Zungqr(m-p, m-p, q, u2, work.Off(itaup2-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantv1t && q > 0 {
			v1t.Set(0, 0, one)
			for j = 2; j <= q; j++ {
				v1t.Set(0, j-1, zero)
				v1t.Set(j-1, 0, zero)
			}
			Zlacpy(Upper, q-1, q-1, x21.Off(0, 1), v1t.Off(1, 1))
			if err = Zunglq(q-1, q-1, q-1, v1t.Off(1, 1), work.Off(itauq1-1), work.Off(iorglq-1), lorglq); err != nil {
				panic(err)
			}
		}

		//        Simultaneously diagonalize X11 and X21.
		if _, err = Zbbcsd(jobu1, jobu2, jobv1t, 'N', NoTrans, m, p, q, theta, rwork.Off(iphi-1), u1, u2, v1t, cdum, rwork.Off(ib11d-1), rwork.Off(ib11e-1), rwork.Off(ib12d-1), rwork.Off(ib12e-1), rwork.Off(ib21d-1), rwork.Off(ib21e-1), rwork.Off(ib22d-1), rwork.Off(ib22e-1), rwork.Off(ibbcsd-1), lbbcsd); err != nil {
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
			Zlapmt(false, m-p, m-p, u2, iwork)
		}
	} else if r == p {
		//        Case 2: R = P
		//
		//        Simultaneously bidiagonalize X11 and X21
		if err = Zunbdb2(m, p, q, x11, x21, theta, rwork.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(iorbdb-1), lorbdb); err != nil {
			panic(err)
		}

		//        Accumulate Householder reflectors
		if wantu1 && p > 0 {
			u1.Set(0, 0, one)
			for j = 2; j <= p; j++ {
				u1.Set(0, j-1, zero)
				u1.Set(j-1, 0, zero)
			}
			Zlacpy(Lower, p-1, p-1, x11.Off(1, 0), u1.Off(1, 1))
			if err = Zungqr(p-1, p-1, p-1, u1.Off(1, 1), work.Off(itaup1-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantu2 && m-p > 0 {
			Zlacpy(Lower, m-p, q, x21, u2)
			if err = Zungqr(m-p, m-p, q, u2, work.Off(itaup2-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantv1t && q > 0 {
			Zlacpy(Upper, p, q, x11, v1t)
			if err = Zunglq(q, q, r, v1t, work.Off(itauq1-1), work.Off(iorglq-1), lorglq); err != nil {
				panic(err)
			}
		}

		//        Simultaneously diagonalize X11 and X21.
		if _, err = Zbbcsd(jobv1t, 'N', jobu1, jobu2, Trans, m, q, p, theta, rwork.Off(iphi-1), v1t, cdum, u1, u2, rwork.Off(ib11d-1), rwork.Off(ib11e-1), rwork.Off(ib12d-1), rwork.Off(ib12e-1), rwork.Off(ib21d-1), rwork.Off(ib21e-1), rwork.Off(ib22d-1), rwork.Off(ib22e-1), rwork.Off(ibbcsd-1), lbbcsd); err != nil {
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
			Zlapmt(false, m-p, m-p, u2, iwork)
		}
	} else if r == m-p {
		//        Case 3: R = M-P
		//
		//        Simultaneously bidiagonalize X11 and X21
		if err = Zunbdb3(m, p, q, x11, x21, theta, rwork.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(iorbdb-1), lorbdb); err != nil {
			panic(err)
		}

		//        Accumulate Householder reflectors
		if wantu1 && p > 0 {
			Zlacpy(Lower, p, q, x11, u1)
			if err = Zungqr(p, p, q, u1, work.Off(itaup1-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantu2 && m-p > 0 {
			u2.Set(0, 0, one)
			for j = 2; j <= m-p; j++ {
				u2.Set(0, j-1, zero)
				u2.Set(j-1, 0, zero)
			}
			Zlacpy(Lower, m-p-1, m-p-1, x21.Off(1, 0), u2.Off(1, 1))
			if err = Zungqr(m-p-1, m-p-1, m-p-1, u2.Off(1, 1), work.Off(itaup2-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantv1t && q > 0 {
			Zlacpy(Upper, m-p, q, x21, v1t)
			if err = Zunglq(q, q, r, v1t, work.Off(itauq1-1), work.Off(iorglq-1), lorglq); err != nil {
				panic(err)
			}
		}

		//        Simultaneously diagonalize X11 and X21.
		if _, err = Zbbcsd('N', jobv1t, jobu2, jobu1, Trans, m, m-q, m-p, theta, rwork.Off(iphi-1), cdum, v1t, u2, u1, rwork.Off(ib11d-1), rwork.Off(ib11e-1), rwork.Off(ib12d-1), rwork.Off(ib12e-1), rwork.Off(ib21d-1), rwork.Off(ib21e-1), rwork.Off(ib22d-1), rwork.Off(ib22e-1), rwork.Off(ibbcsd-1), lbbcsd); err != nil {
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
				Zlapmt(false, p, q, u1, iwork)
			}
			if wantv1t {
				Zlapmr(false, q, q, v1t, iwork)
			}
		}
	} else {
		//        Case 4: R = M-Q
		//
		//        Simultaneously bidiagonalize X11 and X21
		if err = Zunbdb4(m, p, q, x11, x21, theta, rwork.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(iorbdb-1), work.Off(iorbdb+m-1), lorbdb-m); err != nil {
			panic(err)
		}

		//        Accumulate Householder reflectors
		if wantu1 && p > 0 {
			u1.Off(0, 0).CVector().Copy(p, work.Off(iorbdb-1), 1, 1)
			for j = 2; j <= p; j++ {
				u1.Set(0, j-1, zero)
			}
			Zlacpy(Lower, p-1, m-q-1, x11.Off(1, 0), u1.Off(1, 1))
			if err = Zungqr(p, p, m-q, u1, work.Off(itaup1-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantu2 && m-p > 0 {
			u2.Off(0, 0).CVector().Copy(m-p, work.Off(iorbdb+p-1), 1, 1)
			for j = 2; j <= m-p; j++ {
				u2.Set(0, j-1, zero)
			}
			Zlacpy(Lower, m-p-1, m-q-1, x21.Off(1, 0), u2.Off(1, 1))
			if err = Zungqr(m-p, m-p, m-q, u2, work.Off(itaup2-1), work.Off(iorgqr-1), lorgqr); err != nil {
				panic(err)
			}
		}
		if wantv1t && q > 0 {
			Zlacpy(Upper, m-q, q, x21, v1t)
			Zlacpy(Upper, p-(m-q), q-(m-q), x11.Off(m-q, m-q), v1t.Off(m-q, m-q))
			Zlacpy(Upper, -p+q, q-p, x21.Off(m-q, p), v1t.Off(p, p))
			if err = Zunglq(q, q, q, v1t, work.Off(itauq1-1), work.Off(iorglq-1), lorglq); err != nil {
				panic(err)
			}
		}

		//        Simultaneously diagonalize X11 and X21.
		if _, err = Zbbcsd(jobu2, jobu1, 'N', jobv1t, NoTrans, m, m-p, m-q, theta, rwork.Off(iphi-1), u2, u1, cdum, v1t, rwork.Off(ib11d-1), rwork.Off(ib11e-1), rwork.Off(ib12d-1), rwork.Off(ib12e-1), rwork.Off(ib21d-1), rwork.Off(ib21e-1), rwork.Off(ib22d-1), rwork.Off(ib22e-1), rwork.Off(ibbcsd-1), lbbcsd); err != nil {
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
				Zlapmt(false, p, p, u1, iwork)
			}
			if wantv1t {
				Zlapmr(false, p, q, v1t, iwork)
			}
		}
	}

	return
}
