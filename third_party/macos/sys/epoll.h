#pragma once

 int epoll_create1(int flags);
enum
  {
    EPOLL_CLOEXEC = 02000000
#define EPOLL_CLOEXEC EPOLL_CLOEXEC
  };

enum EPOLL_EVENTS {
	EPOLLIN = 1,
	EPOLLPRI = 2,
	EPOLLOUT = 4,
	EPOLLRDNORM = 64,
	EPOLLRDBAND = 128,
	EPOLLWRNORM = 256,
	EPOLLWRBAND = 512,
	EPOLLMSG = 1024,
	EPOLLERR = 8,
	EPOLLHUP = 16,
	EPOLLRDHUP = 8192,
	EPOLLEXCLUSIVE = 268435456,
	EPOLLWAKEUP = 536870912,
	EPOLLONESHOT = 1073741824,
	EPOLLET = 2147483648,
};

typedef union epoll_data
{
  void *ptr;
  int fd;
  uint32_t u32;
  uint64_t u64;
} epoll_data_t;

struct epoll_event
{
  uint32_t events;	/* Epoll events */
  epoll_data_t data;	/* User data variable */
};



/* Valid opcodes ( "op" parameter ) to issue to epoll_ctl().  */
#define EPOLL_CTL_ADD 1	/* Add a file descriptor to the interface.  */
#define EPOLL_CTL_DEL 2	/* Remove a file descriptor from the interface.  */
#define EPOLL_CTL_MOD 3	/* Change file descriptor epoll_event structure.  */

int  epoll_wait (int __epfd, struct epoll_event *__events,
		       int __maxevents, int __timeout);

 int epoll_ctl (int __epfd, int __op, int __fd,
		      struct epoll_event *__event) ;
