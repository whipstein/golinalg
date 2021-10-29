package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhseqr computes the eigenvalues of a Hessenberg matrix H
//    and, optionally, the matrices T and Z from the Schur decomposition
//    H = Z T Z**H, where T is an upper triangular matrix (the
//    Schur form), and Z is the unitary matrix of Schur vectors.
//
//    Optionally Z may be postmultiplied into an input unitary
//    matrix Q so that this routine can give the Schur factorization
//    of a matrix A which has been reduced to the Hessenberg form H
//    by the unitary matrix Q:  A = Q*H*Q**H = (QZ)*T*(QZ)**H.
func Zhseqr(job, compz byte, n, ilo, ihi int, h *mat.CMatrix, w *mat.CVector, z *mat.CMatrix, work *mat.CVector, lwork int) (info int, err error) {
	var initz, lquery, wantt, wantz bool
	var one, zero complex128
	var rzero float64
	var kbot, nl, nmin, ntiny int

	workl := cvf(49)
	hl := cmf(49, 49, opts)

	ntiny = 11

	//     ==== NL allocates some local workspace to help small matrices
	//     .    through a rare ZLAHQR failure.  NL > NTINY = 11 is
	//     .    required and NL <= NMIN = ILAENV(ISPEC=12,...) is recom-
	//     .    mended.  (The default value of NMIN is 75.)  Using NL = 49
	//     .    allows up to six simultaneous shifts and a 16-by-16
	//     .    deflation window.  ====
	nl = 49
	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	rzero = 0.0

	//     ==== Decode and check the input parameters. ====
	wantt = job == 'S'
	initz = compz == 'I'
	wantz = initz || compz == 'V'
	work.Set(0, complex(float64(max(1, n)), rzero))
	lquery = lwork == -1

	if job != 'E' && !wantt {
		err = fmt.Errorf("job != 'E' && !wantt: job='%c'", job)
	} else if compz != 'N' && !wantz {
		err = fmt.Errorf("compz != 'N' && !wantz: compz='%c'", compz)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ilo < 1 || ilo > max(1, n) {
		err = fmt.Errorf("ilo < 1 || ilo > max(1, n): n=%v, ilo=%v", n, ilo)
	} else if ihi < min(ilo, n) || ihi > n {
		err = fmt.Errorf("ihi < min(ilo, n) || ihi > n: n=%v, ilo=%v, ihi=%v", n, ilo, ihi)
	} else if h.Rows < max(1, n) {
		err = fmt.Errorf("h.Rows < max(1, n): h.Rows=%v, n=%v", h.Rows, n)
	} else if z.Rows < 1 || (wantz && z.Rows < max(1, n)) {
		err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < max(1, n)): compz='%c', z.Rows=%v, n=%v", compz, z.Rows, n)
	} else if lwork < max(1, n) && !lquery {
		err = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}

	if err != nil {
		//        ==== Quick return in case of invalid argument. ====
		gltest.Xerbla2("Zhseqr", err)
		return

	} else if n == 0 {
		//        ==== Quick return in case N = 0; nothing to do. ====
		return

	} else if lquery {
		//        ==== Quick return in case of a workspace query ====
		info = Zlaqr0(wantt, wantz, n, ilo, ihi, h, w, ilo, ihi, z, work, lwork)
		//        ==== Ensure reported workspace size is backward-compatible with
		//        .    previous LAPACK versions. ====
		work.Set(0, complex(math.Max(work.GetRe(0), float64(max(1, n))), rzero))
		return

	} else {
		//        ==== copy eigenvalues isolated by ZGEBAL ====
		if ilo > 1 {
			goblas.Zcopy(ilo-1, h.CVector(0, 0, h.Rows+1), w.Off(0, 1))
		}
		if ihi < n {
			goblas.Zcopy(n-ihi, h.CVector(ihi, ihi, h.Rows+1), w.Off(ihi, 1))
		}

		//        ==== Initialize Z, if requested ====
		if initz {
			Zlaset(Full, n, n, zero, one, z)
		}

		//        ==== Quick return if possible ====
		if ilo == ihi {
			w.Set(ilo-1, h.Get(ilo-1, ilo-1))
			return
		}

		//        ==== ZLAHQR/ZLAQR0 crossover point ====
		nmin = Ilaenv(12, "Zhseqr", []byte{job, compz}, n, ilo, ihi, lwork)
		nmin = max(ntiny, nmin)

		//        ==== ZLAQR0 for big matrices; ZLAHQR for small ones ====
		if n > nmin {
			info = Zlaqr0(wantt, wantz, n, ilo, ihi, h, w, ilo, ihi, z, work, lwork)
		} else {
			//           ==== Small matrix ====
			info = Zlahqr(wantt, wantz, n, ilo, ihi, h, w, ilo, ihi, z)

			if info > 0 {
				//              ==== A rare ZLAHQR failure!  ZLAQR0 sometimes succeeds
				//              .    when ZLAHQR fails. ====
				kbot = info

				if n >= nl {
					//                 ==== Larger matrices have enough subdiagonal scratch
					//                 .    space to call ZLAQR0 directly. ====
					info = Zlaqr0(wantt, wantz, n, ilo, kbot, h, w, ilo, ihi, z, work, lwork)

				} else {
					//                 ==== Tiny matrices don't have enough subdiagonal
					//                 .    scratch space to benefit from ZLAQR0.  Hence,
					//                 .    tiny matrices must be copied into a larger
					//                 .    array before calling ZLAQR0. ====
					Zlacpy(Full, n, n, h, hl)
					hl.Set(n, n-1, zero)
					Zlaset(Full, nl, nl-n, zero, zero, hl.Off(0, n))
					info = Zlaqr0(wantt, wantz, nl, ilo, kbot, hl, w, ilo, ihi, z, workl, nl)
					if wantt || info != 0 {
						Zlacpy(Full, n, n, hl, h)
					}
				}
			}
		}

		//        ==== Clear out the trash, if necessary. ====
		if (wantt || info != 0) && n > 2 {
			Zlaset(Lower, n-2, n-2, zero, zero, h.Off(2, 0))
		}

		//        ==== Ensure reported workspace size is backward-compatible with
		//        .    previous LAPACK versions. ====
		work.Set(0, complex(math.Max(float64(max(1, n)), work.GetRe(0)), rzero))
	}

	return
}
