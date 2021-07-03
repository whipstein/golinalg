package golapack

import (
	"github.com/whipstein/golinalg/goblas"
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
// R = minint(P,M-P,Q,M-Q). I1 is a K1-by-K1 identity matrix and I2 is a
// K2-by-K2 identity matrix, where K1 = maxint(Q+P-M,0), K2 = maxint(Q-P,0).
func Zuncsd2by1(jobu1, jobu2, jobv1t byte, m, p, q *int, x11 *mat.CMatrix, ldx11 *int, x21 *mat.CMatrix, ldx21 *int, theta *mat.Vector, u1 *mat.CMatrix, ldu1 *int, u2 *mat.CMatrix, ldu2 *int, v1t *mat.CMatrix, ldv1t *int, work *mat.CVector, lwork *int, rwork *mat.Vector, lrwork *int, iwork *[]int, info *int) {
	var lquery, wantu1, wantu2, wantv1t bool
	var one, zero complex128
	var childinfo, i, ib11d, ib11e, ib12d, ib12e, ib21d, ib21e, ib22d, ib22e, ibbcsd, iorbdb, iorglq, iorgqr, iphi, itaup1, itaup2, itauq1, j, lbbcsd, lorbdb, lorglq, lorglqmin, lorglqopt, lorgqr, lorgqrmin, lorgqropt, lrworkmin, lrworkopt, lworkmin, lworkopt, r int
	dum := vf(1)
	cdum := cmf(1, 1, opts)

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test input arguments
	(*info) = 0
	wantu1 = jobu1 == 'Y'
	wantu2 = jobu2 == 'Y'
	wantv1t = jobv1t == 'Y'
	lquery = (*lwork) == -1

	if (*m) < 0 {
		(*info) = -4
	} else if (*p) < 0 || (*p) > (*m) {
		(*info) = -5
	} else if (*q) < 0 || (*q) > (*m) {
		(*info) = -6
	} else if (*ldx11) < maxint(1, *p) {
		(*info) = -8
	} else if (*ldx21) < maxint(1, (*m)-(*p)) {
		(*info) = -10
	} else if wantu1 && (*ldu1) < maxint(1, *p) {
		(*info) = -13
	} else if wantu2 && (*ldu2) < maxint(1, (*m)-(*p)) {
		(*info) = -15
	} else if wantv1t && (*ldv1t) < maxint(1, *q) {
		(*info) = -17
	}

	r = minint(*p, (*m)-(*p), *q, (*m)-(*q))

	//     Compute workspace
	//
	//       WORK layout:
	//     |-----------------------------------------|
	//     | LWORKOPT (1)                            |
	//     |-----------------------------------------|
	//     | TAUP1 (maxint(1,P))                        |
	//     | TAUP2 (maxint(1,M-P))                      |
	//     | TAUQ1 (maxint(1,Q))                        |
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
	//     | PHI (maxint(1,R-1)) |
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
	if (*info) == 0 {
		iphi = 2
		ib11d = iphi + maxint(1, r-1)
		ib11e = ib11d + maxint(1, r)
		ib12d = ib11e + maxint(1, r-1)
		ib12e = ib12d + maxint(1, r)
		ib21d = ib12e + maxint(1, r-1)
		ib21e = ib21d + maxint(1, r)
		ib22d = ib21e + maxint(1, r-1)
		ib22e = ib22d + maxint(1, r)
		ibbcsd = ib22e + maxint(1, r-1)
		itaup1 = 2
		itaup2 = itaup1 + maxint(1, *p)
		itauq1 = itaup2 + maxint(1, (*m)-(*p))
		iorbdb = itauq1 + maxint(1, *q)
		iorgqr = itauq1 + maxint(1, *q)
		iorglq = itauq1 + maxint(1, *q)
		lorgqrmin = 1
		lorgqropt = 1
		lorglqmin = 1
		lorglqopt = 1
		if r == (*q) {
			Zunbdb1(m, p, q, x11, ldx11, x21, ldx21, theta, dum, cdum.CVector(0, 0), cdum.CVector(0, 0), cdum.CVector(0, 0), work, toPtr(-1), &childinfo)
			lorbdb = int(work.GetRe(0))
			if wantu1 && (*p) > 0 {
				Zungqr(p, p, q, u1, ldu1, cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
				lorgqrmin = maxint(lorgqrmin, *p)
				lorgqropt = maxint(lorgqropt, int(work.GetRe(0)))
			}
			if wantu2 && (*m)-(*p) > 0 {
				Zungqr(toPtr((*m)-(*p)), toPtr((*m)-(*p)), q, u2, ldu2, cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
				lorgqrmin = maxint(lorgqrmin, (*m)-(*p))
				lorgqropt = maxint(lorgqropt, int(work.GetRe(0)))
			}
			if wantv1t && (*q) > 0 {
				Zunglq(toPtr((*q)-1), toPtr((*q)-1), toPtr((*q)-1), v1t, ldv1t, cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
				lorglqmin = maxint(lorglqmin, (*q)-1)
				lorglqopt = maxint(lorglqopt, int(work.GetRe(0)))
			}
			Zbbcsd(jobu1, jobu2, jobv1t, 'N', 'N', m, p, q, theta, dum, u1, ldu1, u2, ldu2, v1t, ldv1t, cdum, func() *int { y := 1; return &y }(), dum, dum, dum, dum, dum, dum, dum, dum, rwork.Off(0), toPtr(-1), &childinfo)
			lbbcsd = int(rwork.Get(0))
		} else if r == (*p) {
			Zunbdb2(m, p, q, x11, ldx11, x21, ldx21, theta, dum, cdum.CVector(0, 0), cdum.CVector(0, 0), cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
			lorbdb = int(work.GetRe(0))
			if wantu1 && (*p) > 0 {
				Zungqr(toPtr((*p)-1), toPtr((*p)-1), toPtr((*p)-1), u1.Off(1, 1), ldu1, cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
				lorgqrmin = maxint(lorgqrmin, (*p)-1)
				lorgqropt = maxint(lorgqropt, int(work.GetRe(0)))
			}
			if wantu2 && (*m)-(*p) > 0 {
				Zungqr(toPtr((*m)-(*p)), toPtr((*m)-(*p)), q, u2, ldu2, cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
				lorgqrmin = maxint(lorgqrmin, (*m)-(*p))
				lorgqropt = maxint(lorgqropt, int(work.GetRe(0)))
			}
			if wantv1t && (*q) > 0 {
				Zunglq(q, q, &r, v1t, ldv1t, cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
				lorglqmin = maxint(lorglqmin, *q)
				lorglqopt = maxint(lorglqopt, int(work.GetRe(0)))
			}
			Zbbcsd(jobv1t, 'N', jobu1, jobu2, 'T', m, q, p, theta, dum, v1t, ldv1t, cdum, func() *int { y := 1; return &y }(), u1, ldu1, u2, ldu2, dum, dum, dum, dum, dum, dum, dum, dum, rwork.Off(0), toPtr(-1), &childinfo)
			lbbcsd = int(rwork.Get(0))
		} else if r == (*m)-(*p) {
			Zunbdb3(m, p, q, x11, ldx11, x21, ldx21, theta, dum, cdum.CVector(0, 0), cdum.CVector(0, 0), cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
			lorbdb = int(work.GetRe(0))
			if wantu1 && (*p) > 0 {
				Zungqr(p, p, q, u1, ldu1, cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
				lorgqrmin = maxint(lorgqrmin, *p)
				lorgqropt = maxint(lorgqropt, int(work.GetRe(0)))
			}
			if wantu2 && (*m)-(*p) > 0 {
				Zungqr(toPtr((*m)-(*p)-1), toPtr((*m)-(*p)-1), toPtr((*m)-(*p)-1), u2.Off(1, 1), ldu2, cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
				lorgqrmin = maxint(lorgqrmin, (*m)-(*p)-1)
				lorgqropt = maxint(lorgqropt, int(work.GetRe(0)))
			}
			if wantv1t && (*q) > 0 {
				Zunglq(q, q, &r, v1t, ldv1t, cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
				lorglqmin = maxint(lorglqmin, *q)
				lorglqopt = maxint(lorglqopt, int(work.GetRe(0)))
			}
			Zbbcsd('N', jobv1t, jobu2, jobu1, 'T', m, toPtr((*m)-(*q)), toPtr((*m)-(*p)), theta, dum, cdum, func() *int { y := 1; return &y }(), v1t, ldv1t, u2, ldu2, u1, ldu1, dum, dum, dum, dum, dum, dum, dum, dum, rwork.Off(0), toPtr(-1), &childinfo)
			lbbcsd = int(rwork.Get(0))
		} else {
			Zunbdb4(m, p, q, x11, ldx11, x21, ldx21, theta, dum, cdum.CVector(0, 0), cdum.CVector(0, 0), cdum.CVector(0, 0), cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
			lorbdb = (*m) + int(work.GetRe(0))
			if wantu1 && (*p) > 0 {
				Zungqr(p, p, toPtr((*m)-(*q)), u1, ldu1, cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
				lorgqrmin = maxint(lorgqrmin, *p)
				lorgqropt = maxint(lorgqropt, int(work.GetRe(0)))
			}
			if wantu2 && (*m)-(*p) > 0 {
				Zungqr(toPtr((*m)-(*p)), toPtr((*m)-(*p)), toPtr((*m)-(*q)), u2, ldu2, cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
				lorgqrmin = maxint(lorgqrmin, (*m)-(*p))
				lorgqropt = maxint(lorgqropt, int(work.GetRe(0)))
			}
			if wantv1t && (*q) > 0 {
				Zunglq(q, q, q, v1t, ldv1t, cdum.CVector(0, 0), work.Off(0), toPtr(-1), &childinfo)
				lorglqmin = maxint(lorglqmin, *q)
				lorglqopt = maxint(lorglqopt, int(work.GetRe(0)))
			}
			Zbbcsd(jobu2, jobu1, 'N', jobv1t, 'N', m, toPtr((*m)-(*p)), toPtr((*m)-(*q)), theta, dum, u2, ldu2, u1, ldu1, cdum, func() *int { y := 1; return &y }(), v1t, ldv1t, dum, dum, dum, dum, dum, dum, dum, dum, rwork.Off(0), toPtr(-1), &childinfo)
			lbbcsd = int(rwork.Get(0))
		}
		lrworkmin = ibbcsd + lbbcsd - 1
		lrworkopt = lrworkmin
		rwork.Set(0, float64(lrworkopt))
		lworkmin = maxint(iorbdb+lorbdb-1, iorgqr+lorgqrmin-1, iorglq+lorglqmin-1)
		lworkopt = maxint(iorbdb+lorbdb-1, iorgqr+lorgqropt-1, iorglq+lorglqopt-1)
		work.SetRe(0, float64(lworkopt))
		if (*lwork) < lworkmin && !lquery {
			(*info) = -19
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNCSD2BY1"), -(*info))
		return
	} else if lquery {
		return
	}
	lorgqr = (*lwork) - iorgqr + 1
	lorglq = (*lwork) - iorglq + 1

	//     Handle four cases separately: R = Q, R = P, R = M-P, and R = M-Q,
	//     in which R = minint(P,M-P,Q,M-Q)
	if r == (*q) {
		//        Case 1: R = Q
		//
		//        Simultaneously bidiagonalize X11 and X21
		Zunbdb1(m, p, q, x11, ldx11, x21, ldx21, theta, rwork.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(iorbdb-1), &lorbdb, &childinfo)

		//        Accumulate Householder reflectors
		if wantu1 && (*p) > 0 {
			Zlacpy('L', p, q, x11, ldx11, u1, ldu1)
			Zungqr(p, p, q, u1, ldu1, work.Off(itaup1-1), work.Off(iorgqr-1), &lorgqr, &childinfo)
		}
		if wantu2 && (*m)-(*p) > 0 {
			Zlacpy('L', toPtr((*m)-(*p)), q, x21, ldx21, u2, ldu2)
			Zungqr(toPtr((*m)-(*p)), toPtr((*m)-(*p)), q, u2, ldu2, work.Off(itaup2-1), work.Off(iorgqr-1), &lorgqr, &childinfo)
		}
		if wantv1t && (*q) > 0 {
			v1t.Set(0, 0, one)
			for j = 2; j <= (*q); j++ {
				v1t.Set(0, j-1, zero)
				v1t.Set(j-1, 0, zero)
			}
			Zlacpy('U', toPtr((*q)-1), toPtr((*q)-1), x21.Off(0, 1), ldx21, v1t.Off(1, 1), ldv1t)
			Zunglq(toPtr((*q)-1), toPtr((*q)-1), toPtr((*q)-1), v1t.Off(1, 1), ldv1t, work.Off(itauq1-1), work.Off(iorglq-1), &lorglq, &childinfo)
		}

		//        Simultaneously diagonalize X11 and X21.
		Zbbcsd(jobu1, jobu2, jobv1t, 'N', 'N', m, p, q, theta, rwork.Off(iphi-1), u1, ldu1, u2, ldu2, v1t, ldv1t, cdum, func() *int { y := 1; return &y }(), rwork.Off(ib11d-1), rwork.Off(ib11e-1), rwork.Off(ib12d-1), rwork.Off(ib12e-1), rwork.Off(ib21d-1), rwork.Off(ib21e-1), rwork.Off(ib22d-1), rwork.Off(ib22e-1), rwork.Off(ibbcsd-1), &lbbcsd, &childinfo)

		//        Permute rows and columns to place zero submatrices in
		//        preferred positions
		if (*q) > 0 && wantu2 {
			for i = 1; i <= (*q); i++ {
				(*iwork)[i-1] = (*m) - (*p) - (*q) + i
			}
			for i = (*q) + 1; i <= (*m)-(*p); i++ {
				(*iwork)[i-1] = i - (*q)
			}
			Zlapmt(false, toPtr((*m)-(*p)), toPtr((*m)-(*p)), u2, ldu2, iwork)
		}
	} else if r == (*p) {
		//        Case 2: R = P
		//
		//        Simultaneously bidiagonalize X11 and X21
		Zunbdb2(m, p, q, x11, ldx11, x21, ldx21, theta, rwork.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(iorbdb-1), &lorbdb, &childinfo)

		//        Accumulate Householder reflectors
		if wantu1 && (*p) > 0 {
			u1.Set(0, 0, one)
			for j = 2; j <= (*p); j++ {
				u1.Set(0, j-1, zero)
				u1.Set(j-1, 0, zero)
			}
			Zlacpy('L', toPtr((*p)-1), toPtr((*p)-1), x11.Off(1, 0), ldx11, u1.Off(1, 1), ldu1)
			Zungqr(toPtr((*p)-1), toPtr((*p)-1), toPtr((*p)-1), u1.Off(1, 1), ldu1, work.Off(itaup1-1), work.Off(iorgqr-1), &lorgqr, &childinfo)
		}
		if wantu2 && (*m)-(*p) > 0 {
			Zlacpy('L', toPtr((*m)-(*p)), q, x21, ldx21, u2, ldu2)
			Zungqr(toPtr((*m)-(*p)), toPtr((*m)-(*p)), q, u2, ldu2, work.Off(itaup2-1), work.Off(iorgqr-1), &lorgqr, &childinfo)
		}
		if wantv1t && (*q) > 0 {
			Zlacpy('U', p, q, x11, ldx11, v1t, ldv1t)
			Zunglq(q, q, &r, v1t, ldv1t, work.Off(itauq1-1), work.Off(iorglq-1), &lorglq, &childinfo)
		}

		//        Simultaneously diagonalize X11 and X21.
		Zbbcsd(jobv1t, 'N', jobu1, jobu2, 'T', m, q, p, theta, rwork.Off(iphi-1), v1t, ldv1t, cdum, func() *int { y := 1; return &y }(), u1, ldu1, u2, ldu2, rwork.Off(ib11d-1), rwork.Off(ib11e-1), rwork.Off(ib12d-1), rwork.Off(ib12e-1), rwork.Off(ib21d-1), rwork.Off(ib21e-1), rwork.Off(ib22d-1), rwork.Off(ib22e-1), rwork.Off(ibbcsd-1), &lbbcsd, &childinfo)

		//        Permute rows and columns to place identity submatrices in
		//        preferred positions
		if (*q) > 0 && wantu2 {
			for i = 1; i <= (*q); i++ {
				(*iwork)[i-1] = (*m) - (*p) - (*q) + i
			}
			for i = (*q) + 1; i <= (*m)-(*p); i++ {
				(*iwork)[i-1] = i - (*q)
			}
			Zlapmt(false, toPtr((*m)-(*p)), toPtr((*m)-(*p)), u2, ldu2, iwork)
		}
	} else if r == (*m)-(*p) {
		//        Case 3: R = M-P
		//
		//        Simultaneously bidiagonalize X11 and X21
		Zunbdb3(m, p, q, x11, ldx11, x21, ldx21, theta, rwork.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(iorbdb-1), &lorbdb, &childinfo)

		//        Accumulate Householder reflectors
		if wantu1 && (*p) > 0 {
			Zlacpy('L', p, q, x11, ldx11, u1, ldu1)
			Zungqr(p, p, q, u1, ldu1, work.Off(itaup1-1), work.Off(iorgqr-1), &lorgqr, &childinfo)
		}
		if wantu2 && (*m)-(*p) > 0 {
			u2.Set(0, 0, one)
			for j = 2; j <= (*m)-(*p); j++ {
				u2.Set(0, j-1, zero)
				u2.Set(j-1, 0, zero)
			}
			Zlacpy('L', toPtr((*m)-(*p)-1), toPtr((*m)-(*p)-1), x21.Off(1, 0), ldx21, u2.Off(1, 1), ldu2)
			Zungqr(toPtr((*m)-(*p)-1), toPtr((*m)-(*p)-1), toPtr((*m)-(*p)-1), u2.Off(1, 1), ldu2, work.Off(itaup2-1), work.Off(iorgqr-1), &lorgqr, &childinfo)
		}
		if wantv1t && (*q) > 0 {
			Zlacpy('U', toPtr((*m)-(*p)), q, x21, ldx21, v1t, ldv1t)
			Zunglq(q, q, &r, v1t, ldv1t, work.Off(itauq1-1), work.Off(iorglq-1), &lorglq, &childinfo)
		}

		//        Simultaneously diagonalize X11 and X21.
		Zbbcsd('N', jobv1t, jobu2, jobu1, 'T', m, toPtr((*m)-(*q)), toPtr((*m)-(*p)), theta, rwork.Off(iphi-1), cdum, func() *int { y := 1; return &y }(), v1t, ldv1t, u2, ldu2, u1, ldu1, rwork.Off(ib11d-1), rwork.Off(ib11e-1), rwork.Off(ib12d-1), rwork.Off(ib12e-1), rwork.Off(ib21d-1), rwork.Off(ib21e-1), rwork.Off(ib22d-1), rwork.Off(ib22e-1), rwork.Off(ibbcsd-1), &lbbcsd, &childinfo)

		//        Permute rows and columns to place identity submatrices in
		//        preferred positions
		if (*q) > r {
			for i = 1; i <= r; i++ {
				(*iwork)[i-1] = (*q) - r + i
			}
			for i = r + 1; i <= (*q); i++ {
				(*iwork)[i-1] = i - r
			}
			if wantu1 {
				Zlapmt(false, p, q, u1, ldu1, iwork)
			}
			if wantv1t {
				Zlapmr(false, q, q, v1t, ldv1t, iwork)
			}
		}
	} else {
		//        Case 4: R = M-Q
		//
		//        Simultaneously bidiagonalize X11 and X21
		Zunbdb4(m, p, q, x11, ldx11, x21, ldx21, theta, rwork.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(iorbdb-1), work.Off(iorbdb+(*m)-1), toPtr(lorbdb-(*m)), &childinfo)

		//        Accumulate Householder reflectors
		if wantu1 && (*p) > 0 {
			goblas.Zcopy(*p, work.Off(iorbdb-1), 1, u1.CVector(0, 0), 1)
			for j = 2; j <= (*p); j++ {
				u1.Set(0, j-1, zero)
			}
			Zlacpy('L', toPtr((*p)-1), toPtr((*m)-(*q)-1), x11.Off(1, 0), ldx11, u1.Off(1, 1), ldu1)
			Zungqr(p, p, toPtr((*m)-(*q)), u1, ldu1, work.Off(itaup1-1), work.Off(iorgqr-1), &lorgqr, &childinfo)
		}
		if wantu2 && (*m)-(*p) > 0 {
			goblas.Zcopy((*m)-(*p), work.Off(iorbdb+(*p)-1), 1, u2.CVector(0, 0), 1)
			for j = 2; j <= (*m)-(*p); j++ {
				u2.Set(0, j-1, zero)
			}
			Zlacpy('L', toPtr((*m)-(*p)-1), toPtr((*m)-(*q)-1), x21.Off(1, 0), ldx21, u2.Off(1, 1), ldu2)
			Zungqr(toPtr((*m)-(*p)), toPtr((*m)-(*p)), toPtr((*m)-(*q)), u2, ldu2, work.Off(itaup2-1), work.Off(iorgqr-1), &lorgqr, &childinfo)
		}
		if wantv1t && (*q) > 0 {
			Zlacpy('U', toPtr((*m)-(*q)), q, x21, ldx21, v1t, ldv1t)
			Zlacpy('U', toPtr((*p)-((*m)-(*q))), toPtr((*q)-((*m)-(*q))), x11.Off((*m)-(*q)+1-1, (*m)-(*q)+1-1), ldx11, v1t.Off((*m)-(*q)+1-1, (*m)-(*q)+1-1), ldv1t)
			Zlacpy('U', toPtr(-(*p)+(*q)), toPtr((*q)-(*p)), x21.Off((*m)-(*q)+1-1, (*p)+1-1), ldx21, v1t.Off((*p)+1-1, (*p)+1-1), ldv1t)
			Zunglq(q, q, q, v1t, ldv1t, work.Off(itauq1-1), work.Off(iorglq-1), &lorglq, &childinfo)
		}

		//        Simultaneously diagonalize X11 and X21.
		Zbbcsd(jobu2, jobu1, 'N', jobv1t, 'N', m, toPtr((*m)-(*p)), toPtr((*m)-(*q)), theta, rwork.Off(iphi-1), u2, ldu2, u1, ldu1, cdum, func() *int { y := 1; return &y }(), v1t, ldv1t, rwork.Off(ib11d-1), rwork.Off(ib11e-1), rwork.Off(ib12d-1), rwork.Off(ib12e-1), rwork.Off(ib21d-1), rwork.Off(ib21e-1), rwork.Off(ib22d-1), rwork.Off(ib22e-1), rwork.Off(ibbcsd-1), &lbbcsd, &childinfo)

		//        Permute rows and columns to place identity submatrices in
		//        preferred positions
		if (*p) > r {
			for i = 1; i <= r; i++ {
				(*iwork)[i-1] = (*p) - r + i
			}
			for i = r + 1; i <= (*p); i++ {
				(*iwork)[i-1] = i - r
			}
			if wantu1 {
				Zlapmt(false, p, p, u1, ldu1, iwork)
			}
			if wantv1t {
				Zlapmr(false, p, q, v1t, ldv1t, iwork)
			}
		}
	}
}
