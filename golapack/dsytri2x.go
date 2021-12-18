package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsytri2x computes the inverse of a real symmetric indefinite matrix
// A using the factorization A = U*D*U**T or A = L*D*L**T computed by
// DSYTRF.
func Dsytri2x(uplo mat.MatUplo, n int, a *mat.Matrix, ipiv *[]int, work *mat.Vector, nb int) (info int, err error) {
	var upper bool
	var ak, akkp1, akp1, d, one, t, u01IJ, u01Ip1J, u11IJ, u11Ip1J, zero float64
	var count, cut, i, invd, ip, j, k, nnb, u11 int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}

	//     Quick return if possible
	if err != nil {
		gltest.Xerbla2("Dsytri2x", err)
		return
	}
	if n == 0 {
		return
	}

	//     Convert A
	//     Workspace got Non-diag elements of D
	if err = Dsyconv(uplo, 'C', n, a, ipiv, work); err != nil {
		panic(err)
	}

	//     Check that the diagonal matrix D is nonsingular.
	if upper {
		//        Upper triangular storage: examine D from bottom to top
		for info = n; info >= 1; info-- {
			if (*ipiv)[info-1] > 0 && a.Get(info-1, info-1) == zero {
				return
			}
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		for info = 1; info <= n; info++ {
			if (*ipiv)[info-1] > 0 && a.Get(info-1, info-1) == zero {
				return
			}
		}
	}
	info = 0

	//  Splitting Workspace
	//     U01 is a block (N,NB+1)
	//     The first element of U01 is in WORK(1,1)
	//     U11 is a block (NB+1,NB+1)
	//     The first element of U11 is in WORK(N+1,1)
	u11 = n
	//     INVD is a block (N,2)
	//     The first element of INVD is in WORK(1,INVD)
	invd = nb + 2
	if upper {
		//        invA = P * inv(U**T)*inv(D)*inv(U)*P**T.
		if info, err = Dtrtri(uplo, Unit, n, a); err != nil {
			panic(err)
		}

		//       inv(D) and inv(D)*inv(U)
		k = 1
		for k <= n {
			if (*ipiv)[k-1] > 0 {
				//           1 x 1 diagonal NNB
				work.Matrix(n+nb+1, opts).Set(k-1, invd-1, one/a.Get(k-1, k-1))
				work.Matrix(n+nb+1, opts).Set(k-1, invd, 0)
				k = k + 1
			} else {
				//           2 x 2 diagonal NNB
				t = work.Matrix(n+nb+1, opts).Get(k, 0)
				ak = a.Get(k-1, k-1) / t
				akp1 = a.Get(k, k) / t
				akkp1 = work.Matrix(n+nb+1, opts).Get(k, 0) / t
				d = t * (ak*akp1 - one)
				work.Matrix(n+nb+1, opts).Set(k-1, invd-1, akp1/d)
				work.Matrix(n+nb+1, opts).Set(k, invd, ak/d)
				work.Matrix(n+nb+1, opts).Set(k-1, invd, -akkp1/d)
				work.Matrix(n+nb+1, opts).Set(k, invd-1, -akkp1/d)
				k = k + 2
			}
		}

		//       inv(U**T) = (inv(U))**T
		//
		//       inv(U**T)*inv(D)*inv(U)
		cut = n
		for cut > 0 {
			nnb = nb
			if cut <= nnb {
				nnb = cut
			} else {
				count = 0
				//             count negative elements,
				for i = cut + 1 - nnb; i <= cut; i++ {
					if (*ipiv)[i-1] < 0 {
						count = count + 1
					}
				}
				//             need a even number for a clear cut
				if count%2 == 1 {
					nnb = nnb + 1
				}
			}
			cut = cut - nnb

			//          U01 Block
			for i = 1; i <= cut; i++ {
				for j = 1; j <= nnb; j++ {
					work.Matrix(n+nb+1, opts).Set(i-1, j-1, a.Get(i-1, cut+j-1))
				}
			}

			//          U11 Block
			for i = 1; i <= nnb; i++ {
				work.Matrix(n+nb+1, opts).Set(u11+i-1, i-1, one)
				for j = 1; j <= i-1; j++ {
					work.Matrix(n+nb+1, opts).Set(u11+i-1, j-1, zero)
				}
				for j = i + 1; j <= nnb; j++ {
					work.Matrix(n+nb+1, opts).Set(u11+i-1, j-1, a.Get(cut+i-1, cut+j-1))
				}
			}

			//          invD*U01
			i = 1
			for i <= cut {
				if (*ipiv)[i-1] > 0 {
					for j = 1; j <= nnb; j++ {
						work.Matrix(n+nb+1, opts).Set(i-1, j-1, work.Matrix(n+nb+1, opts).Get(i-1, invd-1)*work.Matrix(n+nb+1, opts).Get(i-1, j-1))
					}
					i = i + 1
				} else {
					for j = 1; j <= nnb; j++ {
						u01IJ = work.Matrix(n+nb+1, opts).Get(i-1, j-1)
						u01Ip1J = work.Matrix(n+nb+1, opts).Get(i, j-1)
						work.Matrix(n+nb+1, opts).Set(i-1, j-1, work.Matrix(n+nb+1, opts).Get(i-1, invd-1)*u01IJ+work.Matrix(n+nb+1, opts).Get(i-1, invd)*u01Ip1J)
						work.Matrix(n+nb+1, opts).Set(i, j-1, work.Matrix(n+nb+1, opts).Get(i, invd-1)*u01IJ+work.Matrix(n+nb+1, opts).Get(i, invd)*u01Ip1J)
					}
					i = i + 2
				}
			}

			//        invD1*U11
			i = 1
			for i <= nnb {
				if (*ipiv)[cut+i-1] > 0 {
					for j = i; j <= nnb; j++ {
						work.Matrix(n+nb+1, opts).Set(u11+i-1, j-1, work.Matrix(n+nb+1, opts).Get(cut+i-1, invd-1)*work.Matrix(n+nb+1, opts).Get(u11+i-1, j-1))
					}
					i = i + 1
				} else {
					for j = i; j <= nnb; j++ {
						u11IJ = work.Matrix(n+nb+1, opts).Get(u11+i-1, j-1)
						u11Ip1J = work.Matrix(n+nb+1, opts).Get(u11+i, j-1)
						work.Matrix(n+nb+1, opts).Set(u11+i-1, j-1, work.Matrix(n+nb+1, opts).Get(cut+i-1, invd-1)*work.Matrix(n+nb+1, opts).Get(u11+i-1, j-1)+work.Matrix(n+nb+1, opts).Get(cut+i-1, invd)*work.Matrix(n+nb+1, opts).Get(u11+i, j-1))
						work.Matrix(n+nb+1, opts).Set(u11+i, j-1, work.Matrix(n+nb+1, opts).Get(cut+i, invd-1)*u11IJ+work.Matrix(n+nb+1, opts).Get(cut+i, invd)*u11Ip1J)
					}
					i = i + 2
				}
			}

			//       U11**T*invD1*U11->U11
			if err = work.Matrix(n+nb+1, opts).Off(u11, 0).UpdateRows(n+nb+1).Trmm(Left, Upper, Trans, Unit, nnb, nnb, one, a.Off(cut, cut)); err != nil {
				panic(err)
			}

			for i = 1; i <= nnb; i++ {
				for j = i; j <= nnb; j++ {
					a.Set(cut+i-1, cut+j-1, work.Matrix(n+nb+1, opts).Get(u11+i-1, j-1))
				}
			}

			//          U01**T*invD*U01->A(CUT+I,CUT+J)
			if err = work.Matrix(n+nb+1, opts).Off(u11, 0).UpdateRows(n+nb+1).Gemm(Trans, NoTrans, nnb, nnb, cut, one, a.Off(0, cut), work.Matrix(n+nb+1, opts).Off(0, 0).UpdateRows(n+nb+1), zero); err != nil {
				panic(err)
			}

			//        U11 =  U11**T*invD1*U11 + U01**T*invD*U01
			for i = 1; i <= nnb; i++ {
				for j = i; j <= nnb; j++ {
					a.Set(cut+i-1, cut+j-1, a.Get(cut+i-1, cut+j-1)+work.Matrix(n+nb+1, opts).Get(u11+i-1, j-1))
				}
			}

			//        U01 =  U00**T*invD0*U01
			if err = work.Matrix(n+nb+1, opts).Off(0, 0).UpdateRows(n+nb+1).Trmm(Left, uplo, Trans, Unit, cut, nnb, one, a); err != nil {
				panic(err)
			}

			//        Update U01
			for i = 1; i <= cut; i++ {
				for j = 1; j <= nnb; j++ {
					a.Set(i-1, cut+j-1, work.Matrix(n+nb+1, opts).Get(i-1, j-1))
				}
			}

			//      Next Block
		}

		//        Apply PERMUTATIONS P and P**T: P * inv(U**T)*inv(D)*inv(U) *P**T
		i = 1
		for i <= n {
			if (*ipiv)[i-1] > 0 {
				ip = (*ipiv)[i-1]
				if i < ip {
					Dsyswapr(uplo, n, a, i, ip)
				}
				if i > ip {
					Dsyswapr(uplo, n, a, ip, i)
				}
			} else {
				ip = -(*ipiv)[i-1]
				i = i + 1
				if (i - 1) < ip {
					Dsyswapr(uplo, n, a, i-1, ip)
				}
				if (i - 1) > ip {
					Dsyswapr(uplo, n, a, ip, i-1)
				}
			}
			i = i + 1
		}
	} else {

		//        LOWER...
		//
		//        invA = P * inv(U**T)*inv(D)*inv(U)*P**T.
		if info, err = Dtrtri(uplo, Unit, n, a); err != nil {
			panic(err)
		}

		//       inv(D) and inv(D)*inv(U)
		k = n
		for k >= 1 {
			if (*ipiv)[k-1] > 0 {
				//           1 x 1 diagonal NNB
				work.Matrix(n+nb+1, opts).Set(k-1, invd-1, one/a.Get(k-1, k-1))
				work.Matrix(n+nb+1, opts).Set(k-1, invd, 0)
				k = k - 1
			} else {
				//           2 x 2 diagonal NNB
				t = work.Matrix(n+nb+1, opts).Get(k-1-1, 0)
				ak = a.Get(k-1-1, k-1-1) / t
				akp1 = a.Get(k-1, k-1) / t
				akkp1 = work.Matrix(n+nb+1, opts).Get(k-1-1, 0) / t
				d = t * (ak*akp1 - one)
				work.Matrix(n+nb+1, opts).Set(k-1-1, invd-1, akp1/d)
				work.Matrix(n+nb+1, opts).Set(k-1, invd-1, ak/d)
				work.Matrix(n+nb+1, opts).Set(k-1, invd, -akkp1/d)
				work.Matrix(n+nb+1, opts).Set(k-1-1, invd, -akkp1/d)
				k = k - 2
			}
		}

		//       inv(U**T) = (inv(U))**T
		//
		//       inv(U**T)*inv(D)*inv(U)
		cut = 0
		for cut < n {
			nnb = nb
			if cut+nnb > n {
				nnb = n - cut
			} else {
				count = 0
				//             count negative elements,
				for i = cut + 1; i <= cut+nnb; i++ {
					if (*ipiv)[i-1] < 0 {
						count = count + 1
					}
				}
				//             need a even number for a clear cut
				if count%2 == 1 {
					nnb = nnb + 1
				}
			}
			//     L21 Block
			for i = 1; i <= n-cut-nnb; i++ {
				for j = 1; j <= nnb; j++ {
					work.Matrix(n+nb+1, opts).Set(i-1, j-1, a.Get(cut+nnb+i-1, cut+j-1))
				}
			}
			//     L11 Block
			for i = 1; i <= nnb; i++ {
				work.Matrix(n+nb+1, opts).Set(u11+i-1, i-1, one)
				for j = i + 1; j <= nnb; j++ {
					work.Matrix(n+nb+1, opts).Set(u11+i-1, j-1, zero)
				}
				for j = 1; j <= i-1; j++ {
					work.Matrix(n+nb+1, opts).Set(u11+i-1, j-1, a.Get(cut+i-1, cut+j-1))
				}
			}

			//          invD*L21
			i = n - cut - nnb
			for i >= 1 {
				if (*ipiv)[cut+nnb+i-1] > 0 {
					for j = 1; j <= nnb; j++ {
						work.Matrix(n+nb+1, opts).Set(i-1, j-1, work.Matrix(n+nb+1, opts).Get(cut+nnb+i-1, invd-1)*work.Matrix(n+nb+1, opts).Get(i-1, j-1))
					}
					i = i - 1
				} else {
					for j = 1; j <= nnb; j++ {
						u01IJ = work.Matrix(n+nb+1, opts).Get(i-1, j-1)
						u01Ip1J = work.Matrix(n+nb+1, opts).Get(i-1-1, j-1)
						work.Matrix(n+nb+1, opts).Set(i-1, j-1, work.Matrix(n+nb+1, opts).Get(cut+nnb+i-1, invd-1)*u01IJ+work.Matrix(n+nb+1, opts).Get(cut+nnb+i-1, invd)*u01Ip1J)
						work.Matrix(n+nb+1, opts).Set(i-1-1, j-1, work.Matrix(n+nb+1, opts).Get(cut+nnb+i-1-1, invd)*u01IJ+work.Matrix(n+nb+1, opts).Get(cut+nnb+i-1-1, invd-1)*u01Ip1J)
					}
					i = i - 2
				}
			}

			//        invD1*L11
			i = nnb
			for i >= 1 {
				if (*ipiv)[cut+i-1] > 0 {
					for j = 1; j <= nnb; j++ {
						work.Matrix(n+nb+1, opts).Set(u11+i-1, j-1, work.Matrix(n+nb+1, opts).Get(cut+i-1, invd-1)*work.Matrix(n+nb+1, opts).Get(u11+i-1, j-1))
					}
					i = i - 1
				} else {
					for j = 1; j <= nnb; j++ {
						u11IJ = work.Matrix(n+nb+1, opts).Get(u11+i-1, j-1)
						u11Ip1J = work.Matrix(n+nb+1, opts).Get(u11+i-1-1, j-1)
						work.Matrix(n+nb+1, opts).Set(u11+i-1, j-1, work.Matrix(n+nb+1, opts).Get(cut+i-1, invd-1)*work.Matrix(n+nb+1, opts).Get(u11+i-1, j-1)+work.Matrix(n+nb+1, opts).Get(cut+i-1, invd)*u11Ip1J)
						work.Matrix(n+nb+1, opts).Set(u11+i-1-1, j-1, work.Matrix(n+nb+1, opts).Get(cut+i-1-1, invd)*u11IJ+work.Matrix(n+nb+1, opts).Get(cut+i-1-1, invd-1)*u11Ip1J)
					}
					i = i - 2
				}
			}

			//       L11**T*invD1*L11->L11
			if err = work.Matrix(n+nb+1, opts).Off(u11, 0).UpdateRows(n+nb+1).Trmm(Left, uplo, Trans, Unit, nnb, nnb, one, a.Off(cut, cut)); err != nil {
				panic(err)
			}

			for i = 1; i <= nnb; i++ {
				for j = 1; j <= i; j++ {
					a.Set(cut+i-1, cut+j-1, work.Matrix(n+nb+1, opts).Get(u11+i-1, j-1))
				}
			}

			if (cut + nnb) < n {
				//          L21**T*invD2*L21->A(CUT+I,CUT+J)
				if err = work.Matrix(n+nb+1, opts).Off(u11, 0).UpdateRows(n+nb+1).Gemm(Trans, NoTrans, nnb, nnb, n-nnb-cut, one, a.Off(cut+nnb, cut), work.Matrix(n+nb+1, opts).Off(0, 0).UpdateRows(n+nb+1), zero); err != nil {
					panic(err)
				}

				//        L11 =  L11**T*invD1*L11 + U01**T*invD*U01
				for i = 1; i <= nnb; i++ {
					for j = 1; j <= i; j++ {
						a.Set(cut+i-1, cut+j-1, a.Get(cut+i-1, cut+j-1)+work.Matrix(n+nb+1, opts).Get(u11+i-1, j-1))
					}
				}

				//        L01 =  L22**T*invD2*L21
				if err = work.Matrix(n+nb+1, opts).Off(0, 0).UpdateRows(n+nb+1).Trmm(Left, uplo, Trans, Unit, n-nnb-cut, nnb, one, a.Off(cut+nnb, cut+nnb)); err != nil {
					panic(err)
				}

				//      Update L21
				for i = 1; i <= n-cut-nnb; i++ {
					for j = 1; j <= nnb; j++ {
						a.Set(cut+nnb+i-1, cut+j-1, work.Matrix(n+nb+1, opts).Get(i-1, j-1))
					}
				}
			} else {
				//        L11 =  L11**T*invD1*L11
				for i = 1; i <= nnb; i++ {
					for j = 1; j <= i; j++ {
						a.Set(cut+i-1, cut+j-1, work.Matrix(n+nb+1, opts).Get(u11+i-1, j-1))
					}
				}
			}

			//      Next Block
			cut = cut + nnb
		}

		//        Apply PERMUTATIONS P and P**T: P * inv(U**T)*inv(D)*inv(U) *P**T
		i = n
		for i >= 1 {
			if (*ipiv)[i-1] > 0 {
				ip = (*ipiv)[i-1]
				if i < ip {
					Dsyswapr(uplo, n, a, i, ip)
				}
				if i > ip {
					Dsyswapr(uplo, n, a, ip, i)
				}
			} else {
				ip = -(*ipiv)[i-1]
				if i < ip {
					Dsyswapr(uplo, n, a, i, ip)
				}
				if i > ip {
					Dsyswapr(uplo, n, a, ip, i)
				}
				i = i - 1
			}
			i = i - 1
		}
	}

	return
}
